#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <system_error>

namespace lem_converter {

// CSVファイルからのメタデータ（Shift-JISエンコード）
struct LemMetadata {
    int nx;     // 東西方向の点数
    int ny;     // 南北方向の点数
    double y1;  // 区画左下X座標
    double x0;  // 区画左下Y座標
    double y0;  // 区画右上X座標
    double x1;  // 区画右上Y座標
    double dx;  // 東西方向のデータ間隔
    double dy;  // 南北方向のデータ間隔
    int nepsg;  // 平面直角座標系番号
};

class CsvParser {
   public:
    explicit CsvParser(const std::filesystem::path& csv_path);

    // CSVファイルを解析してメタデータを抽出
    [[nodiscard]] std::optional<LemMetadata> parse(std::error_code& ec) const;

   private:
    std::filesystem::path csv_path_;

    // Shift-JISをUTF-8に変換
    static std::string shift_jis_to_utf8(const std::string& shift_jis_str);
};

}  // namespace lem_converter
