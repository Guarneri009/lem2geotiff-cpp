#ifdef USE_CUDA

#    include <iostream>

#    include "cuda/cuda_utils.cuh"
#    include "cuda/lem_parser_cuda.cuh"
#    include "simd_utils.hpp"

namespace lem_converter {
namespace cuda {

__device__ inline int device_fast_parse_int(const char* str, int len) {
    if (len == 5) {
        int result = 0;
        int multiplier = 1;
        bool negative = false;

        for (int i = 4; i >= 0; --i) {
            char c = str[i];
            if (c >= '0' && c <= '9') {
                result += (c - '0') * multiplier;
                multiplier *= 10;
            } else if (c == '-') {
                negative = true;
            }
        }

        return negative ? -result : result;
    }

    int result = 0;
    bool negative = false;
    int i = 0;
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

__global__ void parse_lem_kernel(const char* file_data, const size_t* line_offsets, int nx, int ny,
                                 float nodata_value, float* output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= ny || col >= nx)
        return;

    size_t line_start = line_offsets[row];
    size_t line_end = (row + 1 < ny) ? line_offsets[row + 1] : line_offsets[row] + nx * 5 + 10 + 2;

    // 各行は10文字のプレフィックスで始まり、その後値ごとに5文字
    int data_offset = 10 + col * 5;
    size_t char_pos = line_start + data_offset;

    if (char_pos + 5 > line_end) {
        output[row * nx + col] = nodata_value;
        return;
    }

    int int_value = device_fast_parse_int(&file_data[char_pos], 5);

    // floatに変換
    if (int_value == -1111 || int_value == -9999) {
        output[row * nx + col] = nodata_value;
    } else {
        output[row * nx + col] = static_cast<float>(int_value) * 0.1f;
    }
}

static std::vector<size_t> find_line_offsets_cpu(const char* data, size_t file_size, int ny) {
    return simd::find_line_offsets(data, file_size, static_cast<size_t>(ny));
}

bool LemParserCuda::parse(const char* file_data, size_t file_size, int nx, int ny,
                          float nodata_value, std::vector<float>& output) {
    // CUDAをチェック
    if (!CudaDevice::is_available()) {
        return false;
    }

    // 十分なGPUメモリがあるかチェック
    size_t required_memory = file_size +                  // ファイルデータ
                             (ny + 1) * sizeof(size_t) +  // 行オフセット
                             nx * ny * sizeof(float);     // 出力データ

    if (!CudaDevice::has_sufficient_memory(required_memory)) {
        std::cerr << "CUDA処理に必要なGPUメモリが不足しています" << std::endl;
        return false;
    }

    // CPUで行オフセットを検出
    auto line_offsets = find_line_offsets_cpu(file_data, file_size, ny);
    if (line_offsets.size() < static_cast<size_t>(ny)) {
        std::cerr << "全ての行オフセットの検出に失敗しました" << std::endl;
        return false;
    }

    // デバイスメモリを確保
    DeviceBuffer<char> d_file_data(file_size);
    DeviceBuffer<size_t> d_line_offsets(line_offsets.size());
    DeviceBuffer<float> d_output(nx * ny);

    // データをデバイスにコピー
    if (!d_file_data.copy_from_host(file_data, file_size)) {
        std::cerr << "ファイルデータのデバイスへのコピーに失敗しました" << std::endl;
        return false;
    }

    if (!d_line_offsets.copy_from_host(line_offsets.data(), line_offsets.size())) {
        std::cerr << "行オフセットのデバイスへのコピーに失敗しました" << std::endl;
        return false;
    }

    // カーネル起動パラメータを設定
    dim3 block_size(32, 16);
    dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);

    // カーネルを起動
    parse_lem_kernel<<<grid_size, block_size>>>(d_file_data.get(), d_line_offsets.get(), nx, ny,
                                                nodata_value, d_output.get());

    // カーネル起動チェック
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDAカーネル起動エラー: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    // カーネル完了を待機
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "CUDAカーネル実行エラー: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    output.resize(nx * ny);
    if (!d_output.copy_to_host(output.data(), nx * ny)) {
        std::cerr << "デバイスからの結果コピーに失敗しました" << std::endl;
        return false;
    }

    std::cout << "CUDAを使用してLEMデータの解析に成功しました" << std::endl;
    return true;
}

bool LemParserCuda::should_use_cuda(int nx, int ny) {
    size_t total_elements = static_cast<size_t>(nx) * ny;
    return total_elements >= MIN_ELEMENTS_FOR_CUDA && CudaDevice::is_available();
}

}  // namespace cuda
}  // namespace lem_converter

#endif  // USE_CUDA
