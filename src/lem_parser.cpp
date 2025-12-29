#include "lem_parser.hpp"

#include <tbb/parallel_for.h>

#include <algorithm>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

#include "simd_utils.hpp"

#ifdef USE_CUDA
#    include "cuda/cuda_utils.cuh"
#    include "cuda/lem_parser_cuda.cuh"
#endif

namespace lem_converter {

LemParser::LemParser(const std::filesystem::path& lem_path) : lem_path_(lem_path) {}

inline int fast_parse_int(const char* str, size_t len) {
    int result = 0;
    bool negative = false;
    size_t i = 0;

    // 先頭のスペースをスキップ
    while (i < len && str[i] == ' ')
        ++i;

    if (i < len && str[i] == '-') {
        negative = true;
        ++i;
    }

    while (i < len && str[i] >= '0' && str[i] <= '9') {
        result = result * 10 + (str[i] - '0');
        ++i;
    }

    return negative ? -result : result;
}

std::optional<LemData> LemParser::parse(int nx, int ny, float nodata_value,
                                        std::error_code& ec) const {
    std::ifstream file(lem_path_, std::ios::binary);
    if (!file.is_open()) {
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
        return std::nullopt;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string content(file_size, '\0');
    file.read(&content[0], file_size);
    file.close();

#ifdef USE_CUDA
    // 最初にCUDA実装を試行
    if (cuda::LemParserCuda::should_use_cuda(nx, ny)) {
        std::cout << "CUDAパースを試行中..." << std::endl;

        LemData result;
        result.nx = nx;
        result.ny = ny;

        if (cuda::LemParserCuda::parse(content.data(), file_size, nx, ny, nodata_value,
                                       result.values)) {
            return result;
        }

        // CUDA失敗時、読み込み済みコンテンツでCPUにフォールバック
        std::cout << "CUDAパースに失敗しました。CPU実装にフォールバック" << std::endl;
    } else {
        std::cout << "データサイズがCUDA閾値未満のため、CPU実装を使用" << std::endl;
    }
#endif

    return parse_cpu_from_memory(content.data(), file_size, nx, ny, nodata_value, ec);
}

std::optional<LemData> LemParser::parse_cpu(int nx, int ny, float nodata_value,
                                            std::error_code& ec) const {
    std::ifstream file(lem_path_, std::ios::binary);
    if (!file.is_open()) {
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
        return std::nullopt;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string content(file_size, '\0');
    file.read(&content[0], file_size);
    file.close();

    return parse_cpu_from_memory(content.data(), file_size, nx, ny, nodata_value, ec);
}

std::optional<LemData> LemParser::parse_cpu_from_memory(const char* data, size_t file_size, int nx,
                                                        int ny, float nodata_value,
                                                        std::error_code& ec) const {
    LemData result;
    result.nx = nx;
    result.ny = ny;
    result.values.resize(static_cast<size_t>(nx) * static_cast<size_t>(ny));

    // SIMD高速化検索で全ての行位置を検出
    auto line_starts = simd::find_line_starts(data, file_size, static_cast<size_t>(ny));

    const char* end = data + file_size;

    // TBBを使用して各行を並列パース
    size_t num_rows = std::min(static_cast<size_t>(ny), line_starts.size());

    tbb::parallel_for(size_t(0), num_rows, [&](size_t row_idx) {
        const char* line_start = line_starts[row_idx];
        const char* line_end = line_start;

        // 行末を検出
        while (line_end < end && *line_end != '\n') {
            ++line_end;
        }

        size_t line_len = static_cast<size_t>(line_end - line_start);

        // \r\nを処理
        if (line_len > 0 && line_start[line_len - 1] == '\r') {
            --line_len;
        }

        // 最初の10文字（プレフィックス）をスキップ
        if (line_len >= 10) {
            const char* data_part = line_start + 10;
            size_t data_len = line_len - 10;

            // パース可能な完全な値の数を計算
            int values_to_parse = std::min(nx, static_cast<int>(data_len / 5));

            // 行のSIMD高速化パースを使用
            simd::parse_fixed_width_values(data_part, values_to_parse, &result.values[row_idx * nx],
                                           nodata_value);

            // 残りの部分値を処理
            for (int col_idx = values_to_parse;
                 col_idx < nx && (col_idx * 5) < static_cast<int>(data_len); ++col_idx) {
                int offset = col_idx * 5;
                int int_value =
                    fast_parse_int(data_part + offset, std::min(size_t(5), data_len - offset));

                if (int_value == -1111 || int_value == -9999) {
                    result.at(col_idx, static_cast<int>(row_idx)) = nodata_value;
                } else {
                    result.at(col_idx, static_cast<int>(row_idx)) =
                        static_cast<float>(int_value) * 0.1f;
                }
            }
        }
    });

    if (num_rows != static_cast<size_t>(ny)) {
        std::cerr << "警告: " << num_rows << " 行" << std::endl;
    }

    return result;
}

}  // namespace lem_converter
