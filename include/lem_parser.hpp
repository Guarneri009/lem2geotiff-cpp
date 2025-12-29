#pragma once

#include <filesystem>
#include <optional>
#include <system_error>
#include <vector>

namespace lem_converter {

struct LemData {
    std::vector<float> values;  // 標高値の1次元配列（行優先順序: y * nx + x）
    int nx;                     // 幅
    int ny;                     // 高さ

    // (x, y)の値にアクセスするヘルパー関数
    inline float& at(int x, int y) { return values[y * nx + x]; }

    inline const float& at(int x, int y) const { return values[y * nx + x]; }
};

class LemParser {
   public:
    explicit LemParser(const std::filesystem::path& lem_path);

    // LEMファイルを解析（固定幅形式: 10文字プレフィックス、5文字の値）
    // -1111と-9999をnodata_valueに変換
    // 有効な値に0.1を乗算して0.1m単位からメートルに変換
    [[nodiscard]] std::optional<LemData> parse(int nx, int ny, float nodata_value,
                                               std::error_code& ec) const;

   private:
    std::filesystem::path lem_path_;

    // CPU実装
    [[nodiscard]] std::optional<LemData> parse_cpu(int nx, int ny, float nodata_value,
                                                   std::error_code& ec) const;

    // 事前読み込みメモリバッファからのCPU実装
    [[nodiscard]] std::optional<LemData> parse_cpu_from_memory(const char* data, size_t file_size,
                                                               int nx, int ny, float nodata_value,
                                                               std::error_code& ec) const;
};

}  // namespace lem_converter
