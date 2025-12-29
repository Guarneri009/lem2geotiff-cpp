#include "geotiff_writer.hpp"

#include <geotiff.h>
#include <geotiffio.h>
#include <geo_normalize.h>
#include <geo_tiffp.h>
#include <proj.h>
#include <tiffio.h>
#include <xtiffio.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace lem_converter {

// GDAL NODATA タグ (42113) を libtiff に登録
#define TIFFTAG_GDAL_NODATA 42113

static const TIFFFieldInfo gdal_field_info[] = {
    {TIFFTAG_GDAL_NODATA, -1, -1, TIFF_ASCII, FIELD_CUSTOM, TRUE, FALSE,
     const_cast<char*>("GDALNoDataValue")}};

static TIFFExtendProc parent_extender = nullptr;

static void gdal_tiff_extender(TIFF* tif) {
    TIFFMergeFieldInfo(tif, gdal_field_info,
                       sizeof(gdal_field_info) / sizeof(gdal_field_info[0]));
    if (parent_extender) {
        (*parent_extender)(tif);
    }
}

static void register_gdal_nodata_tag() {
    static bool registered = false;
    if (!registered) {
        parent_extender = TIFFSetTagExtender(gdal_tiff_extender);
        registered = true;
    }
}

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
};

GeoTiffWriter::GeoTiffWriter(Config config) : pImpl(std::make_unique<Impl>(config)) {
    register_gdal_nodata_tag();
}

GeoTiffWriter::~GeoTiffWriter() = default;

GeoTiffWriter::GeoTiffWriter(GeoTiffWriter&&) noexcept = default;
GeoTiffWriter& GeoTiffWriter::operator=(GeoTiffWriter&&) noexcept = default;

bool GeoTiffWriter::create(std::error_code& ec) {
    // 出力ディレクトリが存在しない場合は作成
    if (pImpl->output_path.has_parent_path()) {
        std::filesystem::create_directories(pImpl->output_path.parent_path());
    }

    // TIFFファイルを作成
    TIFF* tif = XTIFFOpen(pImpl->output_path.string().c_str(), "w");
    if (!tif) {
        ec = std::make_error_code(std::errc::io_error);
        return false;
    }

    // 基本TIFFタグを設定
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(pImpl->nx));
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(pImpl->ny));
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

    // タイル形式で圧縮
    const uint32_t tile_width = 256;
    const uint32_t tile_height = 256;
    TIFFSetField(tif, TIFFTAG_TILEWIDTH, tile_width);
    TIFFSetField(tif, TIFFTAG_TILELENGTH, tile_height);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_ADOBE_DEFLATE);

    // GeoTIFFハンドルを取得
    GTIF* gtif = GTIFNew(tif);
    if (!gtif) {
        XTIFFClose(tif);
        ec = std::make_error_code(std::errc::io_error);
        return false;
    }

    // 座標変換パラメータを計算
    double dx_meters = pImpl->metadata.dx * 100.0;
    double dy_meters = pImpl->metadata.dy * 100.0;

    double x_start = pImpl->metadata.x0 / 100.0 + dx_meters / 2.0 / 100.0;
    double y_start = pImpl->metadata.y0 / 100.0 - dy_meters / 2.0 / 100.0;

    // ModelPixelScaleTag: [ScaleX, ScaleY, ScaleZ]
    double pixel_scale[3] = {dx_meters / 100.0, dy_meters / 100.0, 0.0};
    TIFFSetField(tif, GTIFF_PIXELSCALE, 3, pixel_scale);

    // ModelTiepointTag: [I, J, K, X, Y, Z]
    // (0, 0) ピクセルの座標
    double tiepoint[6] = {0.0, 0.0, 0.0, x_start, y_start, 0.0};
    TIFFSetField(tif, GTIFF_TIEPOINTS, 6, tiepoint);

    // GeoTIFFキーを設定
    int epsg = 6668 + pImpl->metadata.nepsg;

    GTIFKeySet(gtif, GTModelTypeGeoKey, TYPE_SHORT, 1, ModelTypeProjected);
    GTIFKeySet(gtif, GTRasterTypeGeoKey, TYPE_SHORT, 1, RasterPixelIsArea);
    GTIFKeySet(gtif, ProjectedCSTypeGeoKey, TYPE_SHORT, 1, epsg);

    // GeoTIFFキーを書き込み
    GTIFWriteKeys(gtif);
    GTIFFree(gtif);

    // GDAL互換のNODATA値タグを設定 (TIFFTAG_GDAL_NODATA = 42113)
    std::string nodata_str = std::to_string(pImpl->nodata_value);
    TIFFSetField(tif, 42113, nodata_str.c_str());

    // タイルデータを書き込み
    std::vector<float> tile_buffer(tile_width * tile_height);

    for (uint32_t ty = 0; ty < static_cast<uint32_t>(pImpl->ny); ty += tile_height) {
        for (uint32_t tx = 0; tx < static_cast<uint32_t>(pImpl->nx); tx += tile_width) {
            // タイルバッファをクリア（NODATA値で埋める）
            std::fill(tile_buffer.begin(), tile_buffer.end(), pImpl->nodata_value);

            // タイル内のデータをコピー
            uint32_t actual_tile_width = std::min(tile_width, static_cast<uint32_t>(pImpl->nx) - tx);
            uint32_t actual_tile_height =
                std::min(tile_height, static_cast<uint32_t>(pImpl->ny) - ty);

            for (uint32_t row = 0; row < actual_tile_height; ++row) {
                for (uint32_t col = 0; col < actual_tile_width; ++col) {
                    size_t src_idx = static_cast<size_t>(ty + row) * pImpl->nx + (tx + col);
                    size_t dst_idx = static_cast<size_t>(row) * tile_width + col;
                    tile_buffer[dst_idx] = pImpl->data[src_idx];
                }
            }

            // タイルを書き込み
            if (TIFFWriteTile(tif, tile_buffer.data(), tx, ty, 0, 0) < 0) {
                XTIFFClose(tif);
                ec = std::make_error_code(std::errc::io_error);
                return false;
            }
        }
    }

    XTIFFClose(tif);

    pImpl->data.clear();
    pImpl->data.shrink_to_fit();

    std::cout << "GeoTIFFを作成しました: " << pImpl->output_path << std::endl;

    return true;
}

// GeoTIFFを読み込むヘルパー関数
struct GeoTiffData {
    std::vector<float> data;
    int width;
    int height;
    double geo_transform[6];  // [x_origin, pixel_width, 0, y_origin, 0, -pixel_height]
    int epsg;
    float nodata_value;
    bool has_nodata;
};

static bool read_geotiff(const std::filesystem::path& path, GeoTiffData& result) {
    TIFF* tif = XTIFFOpen(path.string().c_str(), "r");
    if (!tif) {
        return false;
    }

    uint32_t width, height;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

    result.width = static_cast<int>(width);
    result.height = static_cast<int>(height);

    // GeoTIFF情報を読み込み
    GTIF* gtif = GTIFNew(tif);
    if (gtif) {
        short model_type = 0;
        GTIFKeyGet(gtif, GTModelTypeGeoKey, &model_type, 0, 1);

        short projected_cs = 0;
        if (GTIFKeyGet(gtif, ProjectedCSTypeGeoKey, &projected_cs, 0, 1)) {
            result.epsg = projected_cs;
        } else {
            result.epsg = 0;
        }
        GTIFFree(gtif);
    }

    // PixelScaleとTiepointを読み込み
    double* pixel_scale = nullptr;
    double* tiepoints = nullptr;
    uint16_t count = 0;

    result.geo_transform[0] = 0.0;  // x_origin
    result.geo_transform[1] = 1.0;  // pixel_width
    result.geo_transform[2] = 0.0;  // rotation
    result.geo_transform[3] = 0.0;  // y_origin
    result.geo_transform[4] = 0.0;  // rotation
    result.geo_transform[5] = -1.0; // -pixel_height

    if (TIFFGetField(tif, GTIFF_PIXELSCALE, &count, &pixel_scale) && count >= 2) {
        result.geo_transform[1] = pixel_scale[0];
        result.geo_transform[5] = -pixel_scale[1];
    }

    if (TIFFGetField(tif, GTIFF_TIEPOINTS, &count, &tiepoints) && count >= 6) {
        result.geo_transform[0] = tiepoints[3];
        result.geo_transform[3] = tiepoints[4];
    }

    // NODATA値を読み込み
    result.has_nodata = false;
    result.nodata_value = std::numeric_limits<float>::quiet_NaN();
    char* nodata_str = nullptr;
    if (TIFFGetField(tif, 42113, &nodata_str) && nodata_str) {
        result.nodata_value = static_cast<float>(std::atof(nodata_str));
        result.has_nodata = true;
    }

    // ラスターデータを読み込み
    result.data.resize(static_cast<size_t>(width) * height);

    // タイル形式かストリップ形式かを判定
    if (TIFFIsTiled(tif)) {
        uint32_t tile_width, tile_height;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tile_width);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tile_height);

        std::vector<float> tile_buffer(tile_width * tile_height);

        for (uint32_t ty = 0; ty < height; ty += tile_height) {
            for (uint32_t tx = 0; tx < width; tx += tile_width) {
                if (TIFFReadTile(tif, tile_buffer.data(), tx, ty, 0, 0) < 0) {
                    XTIFFClose(tif);
                    return false;
                }

                uint32_t actual_tile_width = std::min(tile_width, width - tx);
                uint32_t actual_tile_height = std::min(tile_height, height - ty);

                for (uint32_t row = 0; row < actual_tile_height; ++row) {
                    for (uint32_t col = 0; col < actual_tile_width; ++col) {
                        size_t src_idx = static_cast<size_t>(row) * tile_width + col;
                        size_t dst_idx = static_cast<size_t>(ty + row) * width + (tx + col);
                        result.data[dst_idx] = tile_buffer[src_idx];
                    }
                }
            }
        }
    } else {
        // ストリップ形式
        for (uint32_t row = 0; row < height; ++row) {
            if (TIFFReadScanline(tif, result.data.data() + row * width, row, 0) < 0) {
                XTIFFClose(tif);
                return false;
            }
        }
    }

    XTIFFClose(tif);
    return true;
}

// GeoTIFFを書き込むヘルパー関数
static bool write_geotiff(const std::filesystem::path& path, const GeoTiffData& data) {
    TIFF* tif = XTIFFOpen(path.string().c_str(), "w");
    if (!tif) {
        return false;
    }

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(data.width));
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(data.height));
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

    const uint32_t tile_width = 256;
    const uint32_t tile_height = 256;
    TIFFSetField(tif, TIFFTAG_TILEWIDTH, tile_width);
    TIFFSetField(tif, TIFFTAG_TILELENGTH, tile_height);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);

    GTIF* gtif = GTIFNew(tif);
    if (!gtif) {
        XTIFFClose(tif);
        return false;
    }

    double pixel_scale[3] = {data.geo_transform[1], -data.geo_transform[5], 0.0};
    TIFFSetField(tif, GTIFF_PIXELSCALE, 3, pixel_scale);

    double tiepoint[6] = {0.0, 0.0, 0.0, data.geo_transform[0], data.geo_transform[3], 0.0};
    TIFFSetField(tif, GTIFF_TIEPOINTS, 6, tiepoint);

    GTIFKeySet(gtif, GTModelTypeGeoKey, TYPE_SHORT, 1, ModelTypeProjected);
    GTIFKeySet(gtif, GTRasterTypeGeoKey, TYPE_SHORT, 1, RasterPixelIsArea);
    if (data.epsg > 0) {
        GTIFKeySet(gtif, ProjectedCSTypeGeoKey, TYPE_SHORT, 1, data.epsg);
    }

    GTIFWriteKeys(gtif);
    GTIFFree(gtif);

    if (data.has_nodata) {
        std::string nodata_str = std::to_string(data.nodata_value);
        TIFFSetField(tif, 42113, nodata_str.c_str());
    }

    std::vector<float> tile_buffer(tile_width * tile_height);
    float fill_value = data.has_nodata ? data.nodata_value : 0.0f;

    for (uint32_t ty = 0; ty < static_cast<uint32_t>(data.height); ty += tile_height) {
        for (uint32_t tx = 0; tx < static_cast<uint32_t>(data.width); tx += tile_width) {
            std::fill(tile_buffer.begin(), tile_buffer.end(), fill_value);

            uint32_t actual_tile_width =
                std::min(tile_width, static_cast<uint32_t>(data.width) - tx);
            uint32_t actual_tile_height =
                std::min(tile_height, static_cast<uint32_t>(data.height) - ty);

            for (uint32_t row = 0; row < actual_tile_height; ++row) {
                for (uint32_t col = 0; col < actual_tile_width; ++col) {
                    size_t src_idx = static_cast<size_t>(ty + row) * data.width + (tx + col);
                    size_t dst_idx = static_cast<size_t>(row) * tile_width + col;
                    tile_buffer[dst_idx] = data.data[src_idx];
                }
            }

            if (TIFFWriteTile(tif, tile_buffer.data(), tx, ty, 0, 0) < 0) {
                XTIFFClose(tif);
                return false;
            }
        }
    }

    XTIFFClose(tif);
    return true;
}

bool GeoTiffWriter::merge_tiffs(const std::vector<std::filesystem::path>& tiff_files,
                                const std::filesystem::path& output_file, float nodata_value,
                                std::error_code& ec) {
    register_gdal_nodata_tag();

    if (tiff_files.empty()) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    // 全てのGeoTIFFを読み込み、バウンディングボックスを計算
    std::vector<GeoTiffData> datasets;
    datasets.reserve(tiff_files.size());

    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    double pixel_width = 0.0;
    double pixel_height = 0.0;
    int epsg = 0;

    for (const auto& tiff : tiff_files) {
        GeoTiffData data;
        if (!read_geotiff(tiff, data)) {
            std::cerr << "ファイルを開けませんでした: " << tiff << std::endl;
            ec = std::make_error_code(std::errc::io_error);
            return false;
        }

        std::cout << "マージ: " << tiff << std::endl;

        double x0 = data.geo_transform[0];
        double y0 = data.geo_transform[3];
        double x1 = x0 + data.width * data.geo_transform[1];
        double y1 = y0 + data.height * data.geo_transform[5];

        min_x = std::min(min_x, std::min(x0, x1));
        max_x = std::max(max_x, std::max(x0, x1));
        min_y = std::min(min_y, std::min(y0, y1));
        max_y = std::max(max_y, std::max(y0, y1));

        if (pixel_width == 0.0) {
            pixel_width = data.geo_transform[1];
            pixel_height = -data.geo_transform[5];
            epsg = data.epsg;
        }

        datasets.push_back(std::move(data));
    }

    // 出力サイズを計算
    int out_width = static_cast<int>(std::ceil((max_x - min_x) / pixel_width));
    int out_height = static_cast<int>(std::ceil((max_y - min_y) / pixel_height));

    // 出力データを初期化
    GeoTiffData output;
    output.width = out_width;
    output.height = out_height;
    output.data.resize(static_cast<size_t>(out_width) * out_height, nodata_value);
    output.geo_transform[0] = min_x;
    output.geo_transform[1] = pixel_width;
    output.geo_transform[2] = 0.0;
    output.geo_transform[3] = max_y;
    output.geo_transform[4] = 0.0;
    output.geo_transform[5] = -pixel_height;
    output.epsg = epsg;
    output.nodata_value = nodata_value;
    output.has_nodata = true;

    // 各データセットをマージ
    for (const auto& src : datasets) {
        double src_x0 = src.geo_transform[0];
        double src_y0 = src.geo_transform[3];

        int dst_col_start = static_cast<int>(std::round((src_x0 - min_x) / pixel_width));
        int dst_row_start = static_cast<int>(std::round((max_y - src_y0) / pixel_height));

        for (int row = 0; row < src.height; ++row) {
            for (int col = 0; col < src.width; ++col) {
                int dst_row = dst_row_start + row;
                int dst_col = dst_col_start + col;

                if (dst_row >= 0 && dst_row < out_height && dst_col >= 0 && dst_col < out_width) {
                    float value = src.data[static_cast<size_t>(row) * src.width + col];

                    // NODATA以外の値のみコピー
                    if (!src.has_nodata || value != src.nodata_value) {
                        output.data[static_cast<size_t>(dst_row) * out_width + dst_col] = value;
                    }
                }
            }
        }
    }

    // 出力ファイルを書き込み
    if (!write_geotiff(output_file, output)) {
        ec = std::make_error_code(std::errc::io_error);
        return false;
    }

    std::cout << tiff_files.size() << " 個のTIFFを " << output_file << " にマージしました"
              << std::endl;

    return true;
}

bool GeoTiffWriter::reproject_tiff(const std::filesystem::path& input_tiff,
                                   const std::filesystem::path& output_tiff,
                                   const std::string& dst_crs, std::error_code& ec) {
    register_gdal_nodata_tag();

    // 入力GeoTIFFを読み込み
    GeoTiffData src_data;
    if (!read_geotiff(input_tiff, src_data)) {
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
        return false;
    }

    // PROJコンテキストを作成
    PJ_CONTEXT* ctx = proj_context_create();
    if (!ctx) {
        ec = std::make_error_code(std::errc::not_enough_memory);
        return false;
    }

    // ソースCRSを構築
    std::string src_crs = "EPSG:" + std::to_string(src_data.epsg);

    // 同じCRSかどうか確認
    PJ* src_pj = proj_create(ctx, src_crs.c_str());
    PJ* dst_pj = proj_create(ctx, dst_crs.c_str());

    if (!src_pj || !dst_pj) {
        if (src_pj) proj_destroy(src_pj);
        if (dst_pj) proj_destroy(dst_pj);
        proj_context_destroy(ctx);
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    // CRSが同じかチェック
    if (proj_is_equivalent_to(src_pj, dst_pj, PJ_COMP_EQUIVALENT)) {
        std::cout << input_tiff << " は既に " << dst_crs << " スキップします。" << std::endl;
        proj_destroy(src_pj);
        proj_destroy(dst_pj);
        proj_context_destroy(ctx);
        return true;
    }

    // 座標変換オブジェクトを作成
    PJ* transform = proj_create_crs_to_crs(ctx, src_crs.c_str(), dst_crs.c_str(), nullptr);
    if (!transform) {
        proj_destroy(src_pj);
        proj_destroy(dst_pj);
        proj_context_destroy(ctx);
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    // 正規化された変換を取得（東向き・北向き軸順序）
    PJ* norm_transform = proj_normalize_for_visualization(ctx, transform);
    if (norm_transform) {
        proj_destroy(transform);
        transform = norm_transform;
    }

    // ソース画像の四隅を変換してバウンディングボックスを計算
    double src_corners[4][2] = {
        {src_data.geo_transform[0], src_data.geo_transform[3]},  // 左上
        {src_data.geo_transform[0] + src_data.width * src_data.geo_transform[1],
         src_data.geo_transform[3]},  // 右上
        {src_data.geo_transform[0],
         src_data.geo_transform[3] + src_data.height * src_data.geo_transform[5]},  // 左下
        {src_data.geo_transform[0] + src_data.width * src_data.geo_transform[1],
         src_data.geo_transform[3] + src_data.height * src_data.geo_transform[5]}  // 右下
    };

    double dst_min_x = std::numeric_limits<double>::max();
    double dst_max_x = std::numeric_limits<double>::lowest();
    double dst_min_y = std::numeric_limits<double>::max();
    double dst_max_y = std::numeric_limits<double>::lowest();

    for (int i = 0; i < 4; ++i) {
        PJ_COORD src_coord = proj_coord(src_corners[i][0], src_corners[i][1], 0, 0);
        PJ_COORD dst_coord = proj_trans(transform, PJ_FWD, src_coord);

        dst_min_x = std::min(dst_min_x, dst_coord.xy.x);
        dst_max_x = std::max(dst_max_x, dst_coord.xy.x);
        dst_min_y = std::min(dst_min_y, dst_coord.xy.y);
        dst_max_y = std::max(dst_max_y, dst_coord.xy.y);
    }

    // 出力解像度を計算（入力と同様のピクセルサイズを維持）
    double src_pixel_width = src_data.geo_transform[1];
    double src_pixel_height = -src_data.geo_transform[5];
    double dst_pixel_width = src_pixel_width;
    double dst_pixel_height = src_pixel_height;

    int dst_width = static_cast<int>(std::ceil((dst_max_x - dst_min_x) / dst_pixel_width));
    int dst_height = static_cast<int>(std::ceil((dst_max_y - dst_min_y) / dst_pixel_height));

    // 逆変換を作成
    PJ* inv_transform = proj_create_crs_to_crs(ctx, dst_crs.c_str(), src_crs.c_str(), nullptr);
    if (!inv_transform) {
        proj_destroy(transform);
        proj_destroy(src_pj);
        proj_destroy(dst_pj);
        proj_context_destroy(ctx);
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    PJ* norm_inv = proj_normalize_for_visualization(ctx, inv_transform);
    if (norm_inv) {
        proj_destroy(inv_transform);
        inv_transform = norm_inv;
    }

    // 出力データを初期化
    GeoTiffData dst_data;
    dst_data.width = dst_width;
    dst_data.height = dst_height;
    dst_data.data.resize(static_cast<size_t>(dst_width) * dst_height, src_data.nodata_value);
    dst_data.geo_transform[0] = dst_min_x;
    dst_data.geo_transform[1] = dst_pixel_width;
    dst_data.geo_transform[2] = 0.0;
    dst_data.geo_transform[3] = dst_max_y;
    dst_data.geo_transform[4] = 0.0;
    dst_data.geo_transform[5] = -dst_pixel_height;
    dst_data.nodata_value = src_data.nodata_value;
    dst_data.has_nodata = src_data.has_nodata;

    // 出力CRSからEPSGコードを抽出
    dst_data.epsg = 0;
    if (dst_crs.find("EPSG:") == 0) {
        dst_data.epsg = std::stoi(dst_crs.substr(5));
    }

    // 最近傍補間で再投影（逆マッピング）
    for (int dst_row = 0; dst_row < dst_height; ++dst_row) {
        for (int dst_col = 0; dst_col < dst_width; ++dst_col) {
            // 出力ピクセルの中心座標
            double dst_x = dst_min_x + (dst_col + 0.5) * dst_pixel_width;
            double dst_y = dst_max_y - (dst_row + 0.5) * dst_pixel_height;

            // ソースCRSに逆変換
            PJ_COORD dst_coord = proj_coord(dst_x, dst_y, 0, 0);
            PJ_COORD src_coord = proj_trans(inv_transform, PJ_FWD, dst_coord);

            // ソースピクセル座標を計算
            double src_col_f =
                (src_coord.xy.x - src_data.geo_transform[0]) / src_data.geo_transform[1] - 0.5;
            double src_row_f =
                (src_data.geo_transform[3] - src_coord.xy.y) / (-src_data.geo_transform[5]) - 0.5;

            int src_col = static_cast<int>(std::round(src_col_f));
            int src_row = static_cast<int>(std::round(src_row_f));

            // 範囲チェック
            if (src_col >= 0 && src_col < src_data.width && src_row >= 0 &&
                src_row < src_data.height) {
                dst_data.data[static_cast<size_t>(dst_row) * dst_width + dst_col] =
                    src_data.data[static_cast<size_t>(src_row) * src_data.width + src_col];
            }
        }
    }

    // 変換オブジェクトを解放
    proj_destroy(transform);
    proj_destroy(inv_transform);
    proj_destroy(src_pj);
    proj_destroy(dst_pj);
    proj_context_destroy(ctx);

    // 出力ファイルを書き込み
    if (!write_geotiff(output_tiff, dst_data)) {
        ec = std::make_error_code(std::errc::io_error);
        return false;
    }

    std::cout << input_tiff << " を " << dst_crs << " に再投影しました -> " << output_tiff
              << std::endl;

    return true;
}

}  // namespace lem_converter
