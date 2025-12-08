default:
    @just --list

build-cpu:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "CPUビルド中..."
    mkdir -p build-cpu
    cd build-cpu
    cmake .. -DENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j$(nproc)
    echo "CPUビルド完了: build-cpu/lem2geotiff_cpp"

build-gpu:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "GPUビルド中 (CUDA)..."
    export PATH="/usr/local/cuda/bin:$PATH"
    mkdir -p build-gpu
    cd build-gpu
    cmake .. -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    cmake --build . -j$(nproc)
    echo "GPUビルド完了: build-gpu/lem2geotiff_cpp"

build-all: build-cpu build-gpu
    @echo "全ビルド完了！"

build-cpu-debug:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "CPUデバッグビルド中..."
    mkdir -p build-cpu
    cd build-cpu
    cmake .. -DENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug
    cmake --build . -j$(nproc)
    echo "CPUデバッグビルド完了: build-cpu/lem2geotiff_cpp"

build-gpu-debug:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "GPUデバッグビルド中..."
    export PATH="/usr/local/cuda/bin:$PATH"
    mkdir -p build-gpu
    cd build-gpu
    cmake .. -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    cmake --build . -j$(nproc)
    echo "GPUデバッグビルド完了: build-gpu/lem2geotiff_cpp"

clean-cpu:
    rm -rf build-cpu
    @echo "CPUビルドディレクトリを削除しました"

clean-gpu:
    rm -rf build-gpu
    @echo "GPUビルドディレクトリを削除しました"

clean: clean-cpu clean-gpu
    @echo "全ビルドディレクトリを削除しました"

rebuild-cpu: clean-cpu build-cpu

rebuild-gpu: clean-gpu build-gpu

rebuild-all: clean build-all

run-cpu DEM_DIR="./.dem_small" NODATA="-9999":
    time ./build-cpu/lem2geotiff_cpp -d {{DEM_DIR}} -n {{NODATA}}

run-gpu DEM_DIR="./.dem_small" NODATA="-9999":
    time ./build-gpu/lem2geotiff_cpp -d {{DEM_DIR}} -n {{NODATA}}

check-cuda:
    @echo "CUDAの利用可否を確認中..."
    @if command -v nvcc >/dev/null 2>&1; then \
        echo "CUDAコンパイラが見つかりました:"; \
        nvcc --version; \
        echo ""; \
        if command -v nvidia-smi >/dev/null 2>&1; then \
            echo "NVIDIA GPU情報:"; \
            nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader; \
        fi \
    else \
        echo "CUDAコンパイラ (nvcc) が見つかりません。GPUビルドは利用できません。"; \
    fi

format:
    @if command -v clang-format >/dev/null 2>&1; then \
        find src include -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' | xargs clang-format -i; \
        echo "コードのフォーマットが完了しました"; \
    else \
        echo "clang-formatが見つかりません。フォーマットをスキップします。"; \
    fi

config:
    @echo "=== ビルド設定 ==="
    @echo "CPUビルドディレクトリ: build-cpu"
    @echo "GPUビルドディレクトリ: build"
    @echo ""
    @echo "コンパイラ情報:"
    @c++ --version | head -n1
    @echo ""
    @just check-cuda
