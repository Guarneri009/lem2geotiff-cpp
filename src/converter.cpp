#include "converter.hpp"

#include <iostream>

#include "csv_parser.hpp"
#include "geotiff_writer.hpp"
#include "lem_parser.hpp"

namespace lem_converter {

Converter::Converter(Config config) : config_(std::move(config)) {}

bool Converter::run(std::error_code& ec) {
    // LEMパスからCSVパスを導出
    auto csv_path = config_.input_path;
    csv_path.replace_extension(".csv");

    if (!std::filesystem::exists(csv_path)) {
        std::cerr << "CSVファイルが見つかりません: " << csv_path << std::endl;
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
        return false;
    }

    // CSVメタデータを解析
    CsvParser csv_parser(csv_path);
    auto metadata_opt = csv_parser.parse(ec);
    if (!metadata_opt) {
        std::cerr << "CSVの解析に失敗しました: " << csv_path << std::endl;
        return false;
    }
    LemMetadata metadata = *metadata_opt;

    std::cout << "LEMファイルを解析中: " << config_.input_path << std::endl;
    std::cout << "  nx=" << metadata.nx << ", ny=" << metadata.ny << std::endl;

    // LEMファイルを解析
    LemParser lem_parser(config_.input_path);
    // LEMデータをparse CUDAまたはCPU実装で解析
    auto lem_data_opt = lem_parser.parse(metadata.nx, metadata.ny, config_.nodata_value, ec);
    if (!lem_data_opt) {
        std::cerr << "LEMの解析に失敗しました: " << config_.input_path << std::endl;
        return false;
    }
    LemData lem_data = *lem_data_opt;

    std::cout << "GeoTIFFを作成中: " << config_.output_path << std::endl;

    // GeoTIFFを作成
    GeoTiffWriter::Config writer_config;
    writer_config.output_path = config_.output_path;
    writer_config.data = std::move(lem_data.values);
    writer_config.nx = metadata.nx;
    writer_config.ny = metadata.ny;
    writer_config.metadata = metadata;
    writer_config.nodata_value = config_.nodata_value;

    GeoTiffWriter writer(std::move(writer_config));
    if (!writer.create(ec)) {
        std::cerr << "GeoTIFFの作成に失敗しました: " << config_.output_path << std::endl;
        return false;
    }

    return true;
}

}  // namespace lem_converter
