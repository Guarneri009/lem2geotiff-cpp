#include "csv_parser.hpp"

#ifdef _WIN32
#    include <windows.h>
#else
#    include <iconv.h>
#endif

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace lem_converter {

CsvParser::CsvParser(const std::filesystem::path& csv_path) : csv_path_(csv_path) {}

std::string CsvParser::shift_jis_to_utf8(const std::string& shift_jis_str) {
#ifdef _WIN32
    if (shift_jis_str.empty()) {
        return shift_jis_str;
    }

    int wide_len = MultiByteToWideChar(932, 0, shift_jis_str.c_str(),
                                       static_cast<int>(shift_jis_str.size()), nullptr, 0);
    if (wide_len == 0) {
        return shift_jis_str;
    }

    std::wstring wide_str(wide_len, L'\0');
    MultiByteToWideChar(932, 0, shift_jis_str.c_str(), static_cast<int>(shift_jis_str.size()),
                        &wide_str[0], wide_len);

    // UTF-16 -> UTF-8
    int utf8_len =
        WideCharToMultiByte(CP_UTF8, 0, wide_str.c_str(), wide_len, nullptr, 0, nullptr, nullptr);
    if (utf8_len == 0) {
        return shift_jis_str;
    }

    std::string utf8_str(utf8_len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide_str.c_str(), wide_len, &utf8_str[0], utf8_len, nullptr,
                        nullptr);

    return utf8_str;
#else

    iconv_t cd = iconv_open("UTF-8", "SHIFT-JIS");
    if (cd == (iconv_t)-1) {
        return shift_jis_str;
    }

    size_t in_bytes_left = shift_jis_str.size();
    size_t out_bytes_left = in_bytes_left * 4;  // UTF-8は1文字あたり最大4バイト

    std::string result(out_bytes_left, '\0');

    char* in_buf = const_cast<char*>(shift_jis_str.data());
    char* out_buf = &result[0];

    size_t ret = iconv(cd, &in_buf, &in_bytes_left, &out_buf, &out_bytes_left);
    iconv_close(cd);

    if (ret == (size_t)-1) {
        return shift_jis_str;
    }

    result.resize(result.size() - out_bytes_left);
    return result;
#endif
}

std::optional<LemMetadata> CsvParser::parse(std::error_code& ec) const {
    // ファイル全体を一度に読み込む
    std::ifstream file(csv_path_, std::ios::binary);
    if (!file.is_open()) {
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
        return std::nullopt;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string content(file_size, '\0');
    file.read(&content[0], file_size);

    // Shift-JISからUTF-8に変換
    std::string utf8_content = shift_jis_to_utf8(content);

    // CSVを解析（形式: key,value）
    std::map<std::string, std::string> data;
    std::istringstream iss(utf8_content);
    std::string line;

    while (std::getline(iss, line)) {
        if (line.empty())
            continue;

        auto comma_pos = line.find(',');
        if (comma_pos == std::string::npos)
            continue;

        std::string key = line.substr(0, comma_pos);
        std::string value = line.substr(comma_pos + 1);

        // 末尾の空白/改行を削除
        while (!value.empty() &&
               (value.back() == '\r' || value.back() == '\n' || value.back() == ' ')) {
            value.pop_back();
        }

        data[key] = value;
    }

    LemMetadata metadata;

    try {
        metadata.nx = std::stoi(data["東西方向の点数"]);
        metadata.ny = std::stoi(data["南北方向の点数"]);
        metadata.y1 = std::stod(data["区画左下X座標"]);
        metadata.x0 = std::stod(data["区画左下Y座標"]);
        metadata.y0 = std::stod(data["区画右上X座標"]);
        metadata.x1 = std::stod(data["区画右上Y座標"]);
        metadata.dx = std::stod(data["東西方向のデータ間隔"]);
        metadata.dy = std::stod(data["南北方向のデータ間隔"]);
        metadata.nepsg = std::stoi(data["平面直角座標系番号"]);
    } catch (const std::exception& e) {
        std::cerr << "CSVパースエラー: " << e.what() << std::endl;
        ec = std::make_error_code(std::errc::invalid_argument);
        return std::nullopt;
    }

    return metadata;
}

}  // namespace lem_converter
