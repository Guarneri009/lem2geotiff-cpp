#pragma once

#include <filesystem>
#include <string>
#include <system_error>

namespace lem_converter {

class Converter {
   public:
    struct Config {
        std::filesystem::path input_path;   // .lemファイルへのパス
        std::filesystem::path output_path;  // .tifファイルのパス
        float nodata_value = -9999.0f;
    };

    explicit Converter(Config config);

    // 単一の.lemファイルをGeoTIFFに変換
    [[nodiscard]] bool run(std::error_code& ec);

   private:
    Config config_;
};

}  // namespace lem_converter
