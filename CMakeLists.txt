# ============================================================================
# lem2geotiff-cpp CMake設定ファイル
# ============================================================================

cmake_minimum_required(VERSION 3.18)

# CUDAアクセラレーションを有効にするオプション（オプション）
option(ENABLE_CUDA "CUDAアクセラレーションを有効にする（オプション）" OFF)

# ----------------------------------------------------------------------------
# プロジェクト設定
# ----------------------------------------------------------------------------
if(ENABLE_CUDA)
    project(lem2geotiff_cpp VERSION 1.0.0 LANGUAGES CXX CUDA)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA ON")
else()
    project(lem2geotiff_cpp VERSION 1.0.0 LANGUAGES CXX)
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA OFF")
endif()

# ----------------------------------------------------------------------------
# C++標準設定
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# compile_commands.jsonを生成（IDE/エディタ向け）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------------------------------------------------------
# 必須パッケージの検索
# ----------------------------------------------------------------------------
find_package(Threads REQUIRED)

# libtiff, libgeotiff, PROJの検索
find_package(TIFF REQUIRED)

# PROJの検索（CMake config優先、フォールバックでpkg-config）
find_package(PROJ CONFIG QUIET)
if(PROJ_FOUND)
    set(PROJ_TARGET PROJ::proj)
    message(STATUS "PROJ found via CMake config")
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PROJ QUIET IMPORTED_TARGET proj)
        if(PROJ_FOUND)
            set(PROJ_TARGET PkgConfig::PROJ)
            message(STATUS "PROJ found via pkg-config")
        endif()
    endif()
    if(NOT PROJ_FOUND)
        message(FATAL_ERROR "PROJ not found. Please install PROJ library.")
    endif()
endif()

# libgeotiffの検索（CMake config優先、フォールバックでpkg-config）
# vcpkgのlibgeotiffはターゲットではなく変数(GEOTIFF_LIBRARIES, GEOTIFF_INCLUDE_DIR)を提供
find_package(GeoTIFF CONFIG QUIET)
if(GeoTIFF_FOUND)
    # vcpkgの場合: 変数ベース
    if(DEFINED GEOTIFF_LIBRARIES)
        set(GEOTIFF_USE_VARIABLES TRUE)
        message(STATUS "GeoTIFF found via CMake config (variables)")
    else()
        # ターゲットベースの場合
        set(GEOTIFF_TARGET GeoTIFF::GeoTIFF)
        set(GEOTIFF_USE_VARIABLES FALSE)
        message(STATUS "GeoTIFF found via CMake config (target)")
    endif()
else()
    # pkg-configにフォールバック
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GEOTIFF_PKG QUIET IMPORTED_TARGET libgeotiff)
        if(GEOTIFF_PKG_FOUND)
            set(GEOTIFF_TARGET PkgConfig::GEOTIFF_PKG)
            set(GEOTIFF_USE_VARIABLES FALSE)
            set(GeoTIFF_FOUND TRUE)
            message(STATUS "GeoTIFF found via pkg-config")
        endif()
    endif()
    if(NOT GeoTIFF_FOUND)
        message(FATAL_ERROR "GeoTIFF not found. Please install libgeotiff library.")
    endif()
endif()

message(STATUS "TIFF found: ${TIFF_LIBRARIES}")

# ----------------------------------------------------------------------------
# 外部依存関係の取得
# ----------------------------------------------------------------------------
include(FetchContent)

# cxxopts: コマンドライン引数パーサー
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.1.1
)

FetchContent_MakeAvailable(cxxopts)

# ----------------------------------------------------------------------------
# ソースファイルの定義
# ----------------------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/csv_parser.cpp
    src/lem_parser.cpp
    src/geotiff_writer.cpp
    src/converter.cpp
)

# CUDAが利用可能な場合、CUDAソースを追加
if(CUDA_AVAILABLE)
    set(CUDA_SOURCES
        src/cuda/lem_parser_cuda.cu
        src/cuda/cuda_utils.cu
    )
    list(APPEND SOURCES ${CUDA_SOURCES})
endif()

# ----------------------------------------------------------------------------
# ヘッダーファイルの定義
# ----------------------------------------------------------------------------
set(HEADERS
    include/csv_parser.hpp
    include/lem_parser.hpp
    include/geotiff_writer.hpp
    include/converter.hpp
    include/simd_utils.hpp
)

# CUDAが利用可能な場合、CUDAヘッダーを追加
if(CUDA_AVAILABLE)
    set(CUDA_HEADERS
        include/cuda/lem_parser_cuda.cuh
        include/cuda/cuda_utils.cuh
    )
    list(APPEND HEADERS ${CUDA_HEADERS})
endif()

# ----------------------------------------------------------------------------
# 実行ファイルのビルド
# ----------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})

# インクルードディレクトリの設定
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# GeoTIFFインクルードディレクトリ（変数ベースの場合）
if(GEOTIFF_USE_VARIABLES AND DEFINED GEOTIFF_INCLUDE_DIR)
    target_include_directories(${PROJECT_NAME} PRIVATE ${GEOTIFF_INCLUDE_DIR})
endif()

# CUDAインクルードディレクトリを追加
if(CUDA_AVAILABLE)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
endif()

# ----------------------------------------------------------------------------
# ライブラリのリンク
# ----------------------------------------------------------------------------
target_link_libraries(${PROJECT_NAME} PRIVATE
    TIFF::TIFF
    ${PROJ_TARGET}
    cxxopts::cxxopts
    Threads::Threads
)

# GeoTIFFリンク（変数ベースまたはターゲットベース）
if(GEOTIFF_USE_VARIABLES)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${GEOTIFF_LIBRARIES})
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE ${GEOTIFF_TARGET})
endif()

# macOSではiconvを明示的にリンク
if(APPLE)
    find_library(ICONV_LIBRARY iconv)
    if(ICONV_LIBRARY)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${ICONV_LIBRARY})
        message(STATUS "iconv found: ${ICONV_LIBRARY}")
    endif()
endif()

# ----------------------------------------------------------------------------
# C++コンパイルオプション
# ----------------------------------------------------------------------------
if(MSVC)
    # vcpkg x64-windows-static tripletを使用する場合、静的ランタイムライブラリを使用
    # これにより MT_StaticRelease / MTd_StaticDebug でビルドされたライブラリとリンク可能
    # VCPKG_TARGET_TRIPLETまたは環境変数VCPKG_DEFAULT_TRIPLETをチェック
    if(NOT DEFINED VCPKG_TARGET_TRIPLET)
        set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}")
    endif()
    if(VCPKG_TARGET_TRIPLET MATCHES "-static$")
        set_property(TARGET ${PROJECT_NAME} PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        message(STATUS "Using static runtime library for triplet: ${VCPKG_TARGET_TRIPLET}")
    endif()

    # Windows.hのmin/maxマクロを無効化（TBBとの競合を回避）
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)

    # MSVC用コンパイルオプション
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W4
        /permissive-
        /utf-8
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2 /DNDEBUG>
    )
    # AVX2サポート（MSVC）
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
        target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2)
    endif()
else()
    # GCC/Clang用コンパイルオプション
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            -Wall
            -Wextra
            -Wpedantic
            -Wno-unused-parameter
            $<$<CONFIG:Debug>:-g -O0>                       # デバッグビルド
            $<$<CONFIG:Release>:-O3 -DNDEBUG>               # リリースビルド
        >
    )
    # AVX2/FMAサポート（x86_64のみ）
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
        )
    endif()
endif()

# ----------------------------------------------------------------------------
# CUDA設定
# ----------------------------------------------------------------------------
if(CUDA_AVAILABLE)
    # CUDAマクロ定義
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)

    # CUDAターゲットプロパティ
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )

    # CUDAコンパイルオプション
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr         # constexpr制約を緩和
            --use_fast_math                  # 高速数学ライブラリを使用
            -Xcompiler=-Wall
            -Xcompiler=-Wno-unknown-pragmas
            -Xcompiler=-Wno-pedantic
            --diag-suppress=1427,20091       # 特定の警告を抑制
            -Xnvlink=--suppress-stack-size-warning
            $<$<CONFIG:Debug>:-G -g>         # デバッグビルド
            $<$<CONFIG:Release>:-O3 -lineinfo>  # リリースビルド（行情報付き）
        >
    )

    # CUDAアーキテクチャ設定
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_ARCHITECTURES "60;70;75;80;86;89" # Maxwell〜Blackwellはコンパイルできないので除外 100
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # リンカー警告を抑制
    set_property(TARGET ${PROJECT_NAME} APPEND_STRING PROPERTY LINK_FLAGS " -Wno-unused-command-line-argument")

    # CUDAランタイムライブラリをリンク
    target_link_libraries(${PROJECT_NAME} PRIVATE cudart)
endif()

# ----------------------------------------------------------------------------
# TBB（Threading Building Blocks）設定
# ----------------------------------------------------------------------------
find_package(TBB QUIET)
if(TBB_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAS_TBB)
    message(STATUS "TBB検出")
else()
    message(STATUS "TBBが見つかりません")
endif()

# ----------------------------------------------------------------------------
# コンパイラチェック
# ----------------------------------------------------------------------------
include(CheckCXXCompilerFlag)

if(NOT MSVC)
    # C++20サポートの確認
    check_cxx_compiler_flag("-std=c++20" HAS_CXX20)
    if(NOT HAS_CXX20)
        message(FATAL_ERROR "C++20サポートが必要です")
    endif()

    # GCC固有の設定
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0")
            message(FATAL_ERROR "GCC 11以上が必要です")
        endif()

        target_compile_options(${PROJECT_NAME} PRIVATE
            -fconcepts
        )
    endif()
endif()

