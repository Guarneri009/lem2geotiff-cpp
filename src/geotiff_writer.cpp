#include "geotiff_writer.hpp"

#include <cpl_conv.h>
#include <gdal_priv.h>
#include <gdal_utils.h>
#include <gdalwarper.h>
#include <ogr_spatialref.h>

#include <algorithm>
#include <cmath>
#include <execution>
#include <iostream>
#include <numeric>

namespace lem_converter {

class GeoTiffWriter::Impl {
   public:
    explicit Impl(const Config& config)
        : output_path(config.output_path),
          data(config.data),
          nx(config.nx),
          ny(config.ny),
          metadata(config.metadata),
          nodata_value(config.nodata_value) {}

    std::filesystem::path output_path;
    std::vector<float> data;
    int nx;
    int ny;
    LemMetadata metadata;
    float nodata_value;
    GDALDataset* dataset = nullptr;
};

GeoTiffWriter::GeoTiffWriter(Config config) : pImpl(std::make_unique<Impl>(config)) {
    GDALAllRegister();
}

GeoTiffWriter::~GeoTiffWriter() {
    if (pImpl && pImpl->dataset) {
        GDALClose(pImpl->dataset);
    }
}

GeoTiffWriter::GeoTiffWriter(GeoTiffWriter&&) noexcept = default;
GeoTiffWriter& GeoTiffWriter::operator=(GeoTiffWriter&&) noexcept = default;

bool GeoTiffWriter::create(std::error_code& ec) {
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver) {
        ec = std::make_error_code(std::errc::not_supported);
        return false;
    }

    // 出力ディレクトリが存在しない場合は作成
    if (pImpl->output_path.has_parent_path()) {
        std::filesystem::create_directories(pImpl->output_path.parent_path());
    }

    char** options = nullptr;
    options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");
    options = CSLSetNameValue(options, "TILED", "YES");

    pImpl->dataset = driver->Create(pImpl->output_path.string().c_str(), pImpl->nx, pImpl->ny, 1,
                                    GDT_Float32, options);

    CSLDestroy(options);

    if (!pImpl->dataset) {
        ec = std::make_error_code(std::errc::io_error);
        return false;
    }

    double dx_meters = pImpl->metadata.dx * 100.0;
    double dy_meters = pImpl->metadata.dy * 100.0;

    double x_start = pImpl->metadata.x0 / 100.0 + dx_meters / 2.0 / 100.0;
    double y_start = pImpl->metadata.y0 / 100.0 - dy_meters / 2.0 / 100.0;

    // GeoTransform: [左上X, ピクセル幅, 回転, 左上Y, 回転, -ピクセル高さ]
    double geo_transform[6];
    geo_transform[0] = x_start;             // 左上X
    geo_transform[1] = dx_meters / 100.0;   // ピクセル幅
    geo_transform[2] = 0.0;                 // 回転
    geo_transform[3] = y_start;             // 左上Y
    geo_transform[4] = 0.0;                 // 回転
    geo_transform[5] = -dy_meters / 100.0;  // -ピクセル高さ（北向きのため負）

    pImpl->dataset->SetGeoTransform(geo_transform);

    // 投影法を設定 - EPSG:6668+nepsg（日本の平面直角座標系）
    int epsg = 6668 + pImpl->metadata.nepsg;
    OGRSpatialReference srs;
    srs.importFromEPSG(epsg);

    char* wkt = nullptr;
    srs.exportToWkt(&wkt);
    pImpl->dataset->SetProjection(wkt);
    CPLFree(wkt);

    CPLErr err = pImpl->dataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, pImpl->nx, pImpl->ny,
                                                            pImpl->data.data(), pImpl->nx,
                                                            pImpl->ny, GDT_Float32, 0, 0);

    if (err != CE_None) {
        std::cerr << "ラスターデータの書き込みに失敗しました" << std::endl;
        GDALClose(pImpl->dataset);
        pImpl->dataset = nullptr;
        return false;
    }

    pImpl->dataset->GetRasterBand(1)->SetNoDataValue(pImpl->nodata_value);

    GDALClose(pImpl->dataset);
    pImpl->dataset = nullptr;

    pImpl->data.clear();
    pImpl->data.shrink_to_fit();

    std::cout << "GeoTIFFを作成しました: " << pImpl->output_path << std::endl;

    return true;
}

bool GeoTiffWriter::merge_tiffs(const std::vector<std::filesystem::path>& tiff_files,
                                const std::filesystem::path& output_file, float nodata_value,
                                std::error_code& ec) {
    if (tiff_files.empty()) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    // 全てのソースデータセットを開く
    std::vector<GDALDatasetH> src_datasets;
    for (const auto& tiff : tiff_files) {
        GDALDatasetH ds = GDALOpen(tiff.string().c_str(), GA_ReadOnly);
        if (!ds) {
            std::cerr << "ファイルを開けませんでした: " << tiff << std::endl;
            for (auto* ds_ptr : src_datasets) {
                GDALClose(ds_ptr);
            }
            ec = std::make_error_code(std::errc::io_error);
            return false;
        }
        src_datasets.push_back(ds);
        std::cout << "マージ: " << tiff << std::endl;
    }

    char** argv = nullptr;
    argv = CSLAddString(argv, "-of");
    argv = CSLAddString(argv, "GTiff");

    // マルチスレッドマージを有効化
    argv = CSLAddString(argv, "-wo");
    argv = CSLAddString(argv, "NUM_THREADS=ALL_CPUS");

    // warpメモリ制限を増加（マージ操作用に512MB）
    argv = CSLAddString(argv, "-wm");
    argv = CSLAddString(argv, "512");

    // 出力形式を最適化
    argv = CSLAddString(argv, "-co");
    argv = CSLAddString(argv, "COMPRESS=LZW");
    argv = CSLAddString(argv, "-co");
    argv = CSLAddString(argv, "TILED=YES");
    argv = CSLAddString(argv, "-co");
    argv = CSLAddString(argv, "BLOCKXSIZE=256");
    argv = CSLAddString(argv, "-co");
    argv = CSLAddString(argv, "BLOCKYSIZE=256");

    GDALWarpAppOptions* warp_opts = GDALWarpAppOptionsNew(argv, nullptr);
    CSLDestroy(argv);

    // warp/マージを実行
    int error = 0;
    GDALDatasetH merged_ds =
        GDALWarp(output_file.string().c_str(), nullptr, static_cast<int>(src_datasets.size()),
                 src_datasets.data(), warp_opts, &error);

    GDALWarpAppOptionsFree(warp_opts);

    // ソースデータセットを閉じる
    for (auto* ds : src_datasets) {
        GDALClose(ds);
    }

    if (!merged_ds || error != 0) {
        if (merged_ds)
            GDALClose(merged_ds);
        ec = std::make_error_code(std::errc::operation_canceled);
        return false;
    }

    GDALClose(merged_ds);
    std::cout << tiff_files.size() << " 個のTIFFを " << output_file << " にマージしました"
              << std::endl;

    return true;
}

bool GeoTiffWriter::reproject_tiff(const std::filesystem::path& input_tiff,
                                   const std::filesystem::path& output_tiff,
                                   const std::string& dst_crs, std::error_code& ec) {
    GDALDataset* src_ds =
        static_cast<GDALDataset*>(GDALOpen(input_tiff.string().c_str(), GA_ReadOnly));

    if (!src_ds) {
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
        return false;
    }

    // ソースCRSを取得
    const char* src_wkt = src_ds->GetProjectionRef();
    OGRSpatialReference src_srs;
    src_srs.importFromWkt(src_wkt);

    OGRSpatialReference dst_srs;
    dst_srs.SetFromUserInput(dst_crs.c_str());

    // 既に対象CRSかどうか確認
    if (src_srs.IsSame(&dst_srs)) {
        std::cout << input_tiff << " は既に " << dst_crs << " スキップします。" << std::endl;
        GDALClose(src_ds);
        return true;
    }

    // パフォーマンス最適化warpオプションを設定
    char** argv = nullptr;
    argv = CSLAddString(argv, "-t_srs");
    argv = CSLAddString(argv, dst_crs.c_str());
    argv = CSLAddString(argv, "-r");
    argv = CSLAddString(argv, "near");  // 最近傍法（標高データで最速）
    argv = CSLAddString(argv, "-of");
    argv = CSLAddString(argv, "GTiff");

    // マルチスレッドwarpを有効化（全CPUコアを使用）
    argv = CSLAddString(argv, "-wo");
    argv = CSLAddString(argv, "NUM_THREADS=ALL_CPUS");

    // パフォーマンス向上のためwarpメモリ制限を増加（256MB）
    argv = CSLAddString(argv, "-wm");
    argv = CSLAddString(argv, "256");

    // 出力に圧縮を使用（LZWはDEMに対して高速で効果的）
    argv = CSLAddString(argv, "-co");
    argv = CSLAddString(argv, "COMPRESS=LZW");
    argv = CSLAddString(argv, "-co");
    argv = CSLAddString(argv, "TILED=YES");
    argv = CSLAddString(argv, "-co");
    argv = CSLAddString(argv, "BLOCKXSIZE=256");
    argv = CSLAddString(argv, "-co");
    argv = CSLAddString(argv, "BLOCKYSIZE=256");

    GDALWarpAppOptions* warp_opts = GDALWarpAppOptionsNew(argv, nullptr);
    CSLDestroy(argv);

    GDALDatasetH src_datasets[1] = {static_cast<GDALDatasetH>(src_ds)};

    int error = 0;
    GDALDatasetH warped_ds =
        GDALWarp(output_tiff.string().c_str(), nullptr, 1, src_datasets, warp_opts, &error);

    GDALWarpAppOptionsFree(warp_opts);
    GDALClose(src_ds);

    if (!warped_ds || error != 0) {
        if (warped_ds)
            GDALClose(warped_ds);
        ec = std::make_error_code(std::errc::operation_canceled);
        return false;
    }

    GDALClose(warped_ds);
    std::cout << input_tiff << " を " << dst_crs << " に再投影しました -> " << output_tiff
              << std::endl;

    return true;
}

}  // namespace lem_converter
