#ifdef _WIN32
#include <windows.h>
#endif

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for_each.h>

#include <algorithm>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

#include "converter.hpp"
#include "geotiff_writer.hpp"

namespace fs = std::filesystem;
using namespace lem_converter;

std::mutex cout_mutex;

// 単一のLEMファイルを処理する
bool process_lem_file(const fs::path& lem_file, const fs::path& output_dir, float nodata_value) {
    std::error_code ec;

    // 出力ファイル名を作成
    std::string output_filename = lem_file.stem().string() + ".tif";
    fs::path output_path = output_dir / output_filename;

    Converter::Config config;
    config.input_path = lem_file;
    config.output_path = output_path;
    config.nodata_value = nodata_value;

    Converter converter(std::move(config));
    bool success = converter.run(ec);

    if (!success) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "変換に失敗: " << lem_file << std::endl;
        return false;
    }

    return true;
}

// ディレクトリ内の全ての.lemファイルを検索する
std::vector<fs::path> find_lem_files(const fs::path& directory) {
    std::vector<fs::path> lem_files;

    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        return lem_files;
    }

    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".lem") {
            lem_files.push_back(entry.path());
        }
    }

    return lem_files;
}

void process_directory_tree(const fs::path& parent_dir, float nodata_value, int workers) {
    // 全てのディレクトリ（ファイルを含むディレクトリ）を検索
    struct DirectoryInfo {
        fs::path directory_path;
        std::vector<fs::path> lem_files;
    };

    std::vector<DirectoryInfo> leaf_directories;

    // まず親ディレクトリ自体をチェック
    {
        std::vector<fs::path> lem_files;
        for (const auto& file_entry : fs::directory_iterator(parent_dir)) {
            if (file_entry.is_regular_file() && file_entry.path().extension() == ".lem") {
                lem_files.push_back(file_entry.path());
            }
        }
        if (!lem_files.empty()) {
            leaf_directories.push_back({parent_dir, lem_files});
        }
    }

    // 次にサブディレクトリを検索
    for (const auto& entry : fs::recursive_directory_iterator(parent_dir)) {
        if (fs::is_directory(entry.path())) {
            std::vector<fs::path> lem_files;

            // このディレクトリ内の.lemファイルを取得
            for (const auto& file_entry : fs::directory_iterator(entry.path())) {
                if (file_entry.is_regular_file() && file_entry.path().extension() == ".lem") {
                    lem_files.push_back(file_entry.path());
                }
            }

            if (!lem_files.empty()) {
                leaf_directories.push_back({entry.path(), lem_files});
            }
        }
    }

    // 各ディレクトリを処理
    for (const auto& dir_info : leaf_directories) {
        std::cout << "\n=== ディレクトリを処理中: " << dir_info.directory_path
                  << " ===" << std::endl;
        std::cout << dir_info.lem_files.size() << " 個の.lemファイルを検出" << std::endl;

        // ディレクトリパスに基づいて出力ファイル名を作成
        // 末尾のスラッシュを除去するためlexically_normal()で正規化
        fs::path normalized_dir = dir_info.directory_path.lexically_normal();
        std::string relative_path = fs::relative(normalized_dir, parent_dir).string();
        std::replace(relative_path.begin(), relative_path.end(), '/', '_');
        // 親ディレクトリ自体の場合（relative_pathが"."または空）はディレクトリ名を使用
        if (relative_path == "." || relative_path.empty()) {
            relative_path = normalized_dir.filename().string();
            // それでも空の場合は親ディレクトリ名を取得
            if (relative_path.empty()) {
                relative_path = normalized_dir.parent_path().filename().string();
            }
        }
        std::string output_name = relative_path + ".tif";
        std::string merged_temp = relative_path + "_merged_temp.tif";

        // ./outディレクトリを作成
        fs::path out_dir = "./out";
        fs::create_directories(out_dir);

        // 全ての.lemファイルを並列で.tifに変換（TBB enumerable_thread_specific使用）
        tbb::enumerable_thread_specific<std::vector<fs::path>> local_converted;

        tbb::parallel_for_each(
            dir_info.lem_files.begin(), dir_info.lem_files.end(), [&](const fs::path& lem_file) {
                if (process_lem_file(lem_file, out_dir, nodata_value)) {
                    fs::path tif_path = out_dir / (lem_file.stem().string() + ".tif");
                    local_converted.local().push_back(tif_path);
                }
            });

        // スレッドローカル結果をマージ
        std::vector<fs::path> converted_files;
        for (const auto& local : local_converted) {
            converted_files.insert(converted_files.end(), local.begin(), local.end());
        }

        std::cout << converted_files.size() << " 個のファイルを正常に変換" << std::endl;

        if (converted_files.empty()) {
            std::cerr << "正常に変換されたファイルがありません！" << std::endl;
            continue;
        }

        // 個別のTIFFを並列で再投影（TBB enumerable_thread_specific使用）
        std::cout << converted_files.size() << " 個のTIFFをEPSG:3857に再投影中..." << std::endl;
        tbb::enumerable_thread_specific<std::vector<fs::path>> local_reprojected;

        tbb::parallel_for_each(
            converted_files.begin(), converted_files.end(), [&](const auto& tiff_path) {
                fs::path reprojected_path = tiff_path.string() + "_3857.tif";
                std::error_code ec;

                if (GeoTiffWriter::reproject_tiff(tiff_path, reprojected_path, "EPSG:3857", ec)) {
                    local_reprojected.local().push_back(reprojected_path);
                } else {
                    std::cerr << "再投影に失敗しました: " << tiff_path << std::endl;
                }
            });

        // スレッドローカル結果をマージ
        std::vector<fs::path> reprojected_files;
        for (const auto& local : local_reprojected) {
            reprojected_files.insert(reprojected_files.end(), local.begin(), local.end());
        }

        if (reprojected_files.empty()) {
            std::cerr << "正常に再投影されたファイルがありません！" << std::endl;
            continue;
        }

        std::cout << reprojected_files.size() << " 個のファイルを正常に再投影しました" << std::endl;

        // 再投影済みの全TIFFを一度にマージ
        std::cout << reprojected_files.size() << " 個の再投影済みTIFFをマージ中..." << std::endl;
        fs::path final_output = output_name;
        std::error_code ec;

        if (!GeoTiffWriter::merge_tiffs(reprojected_files, final_output, nodata_value, ec)) {
            std::cerr << "TIFFのマージに失敗しました" << std::endl;
            continue;
        }

        std::cout << "全てのファイルを " << final_output << " にマージ" << std::endl;

        // 一時ファイルをクリーンアップ
        for (const auto& file : converted_files) {
            fs::remove(file);  // 元の未変換GeoTIFFを削除
        }
        for (const auto& file : reprojected_files) {
            fs::remove(file);  // 再投影済みGeoTIFFを削除（マージ済み）
        }

        std::cout << "最終出力: " << final_output << std::endl;
    }
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    // Windowsコンソール出力をUTF-8に設定
    SetConsoleOutputCP(CP_UTF8);
#endif

    cxxopts::Options options("lem2geotiff_cpp", "LEMファイルをGeoTIFFに変換");

    options.add_options()("d,directory", "LEMファイルを含む入力ディレクトリ",
                          cxxopts::value<std::string>())(
        "n,nodata", "NoData値（デフォルト: -9999）",
        cxxopts::value<float>()->default_value("-9999"))(
        "w,workers", "並列ワーカー数（デフォルト: 4）", cxxopts::value<int>()->default_value("4"))(
        "h,help", "使用方法を表示");

    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("directory")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string input_dir = result["directory"].as<std::string>();
    float nodata_value = result["nodata"].as<float>();
    int workers = result["workers"].as<int>();

    // 末尾のスラッシュを除去するためlexically_normal()で正規化
    fs::path parent_dir = fs::path(input_dir).lexically_normal();

    if (!fs::exists(parent_dir) || !fs::is_directory(parent_dir)) {
        std::cerr << "エラー: ディレクトリが存在しません: " << parent_dir << std::endl;
        return 1;
    }

    // outディレクトリをクリアして作成
    fs::path out_dir = "./out";
    if (fs::exists(out_dir)) {
        fs::remove_all(out_dir);
        std::cout << "既存の./outディレクトリを削除" << std::endl;
    }
    fs::create_directories(out_dir);
    std::cout << "./outディレクトリを作成" << std::endl;

    std::cout << "\n=== LEMからGeoTIFFへの変換を開始 ===" << std::endl;
    std::cout << "入力ディレクトリ: " << parent_dir << std::endl;
    std::cout << "NoData値: " << nodata_value << std::endl;
    std::cout << "ワーカー数: " << workers << std::endl;

    process_directory_tree(parent_dir, nodata_value, workers);

    std::cout << "\n=== 変換完了 ===" << std::endl;

    return 0;
}
