#pragma once

#include <filesystem>
#include <memory>
#include <system_error>
#include <vector>

#include "csv_parser.hpp"

namespace lem_converter {

class GeoTiffWriter {
   public:
    struct Config {
        std::filesystem::path output_path;
        std::vector<float> data;  // 行優先順序の1次元配列 (y * nx + x)
        int nx;
        int ny;
        LemMetadata metadata;
        float nodata_value;
    };

    explicit GeoTiffWriter(Config config);
    ~GeoTiffWriter();

    GeoTiffWriter(const GeoTiffWriter&) = delete;
    GeoTiffWriter& operator=(const GeoTiffWriter&) = delete;
    GeoTiffWriter(GeoTiffWriter&&) noexcept;
    GeoTiffWriter& operator=(GeoTiffWriter&&) noexcept;

    // 適切な座標系でGeoTIFFを作成 (EPSG:6668+nepsg)
    [[nodiscard]] bool create(std::error_code& ec);

    static bool merge_tiffs(const std::vector<std::filesystem::path>& tiff_files,
                            const std::filesystem::path& output_file, float nodata_value,
                            std::error_code& ec);

    static bool reproject_tiff(const std::filesystem::path& input_tiff,
                               const std::filesystem::path& output_tiff, const std::string& dst_crs,
                               std::error_code& ec);

   private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

}  // namespace lem_converter
