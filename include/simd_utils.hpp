#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#    define HAS_AVX2 1
#    include <immintrin.h>
#else
#    define HAS_AVX2 0
#endif

#ifdef _MSC_VER
#    include <intrin.h>
#endif

namespace lem_converter {
namespace simd {

inline int ctz(uint32_t mask) {
#ifdef _MSC_VER
    unsigned long index;
    _BitScanForward(&index, mask);
    return static_cast<int>(index);
#else
    return __builtin_ctz(mask);
#endif
}

// AVX2 SIMDを使用してバッファ内の全ての改行位置を検出
inline std::vector<const char*> find_line_starts(const char* data, size_t size,
                                                 size_t max_lines = 0) {
    std::vector<const char*> line_starts;
    line_starts.reserve(max_lines > 0 ? max_lines + 1 : size / 50);  // 1行あたり約50文字と推定

    const char* end = data + size;
    const char* pos = data;

    // 最初の行は常に位置0から開始
    line_starts.push_back(pos);

#if HAS_AVX2
    // AVX2最適化パス: 一度に32バイトを処理
    const __m256i newline_mask = _mm256_set1_epi8('\n');

    while (pos + 32 <= end) {
        // 32バイトをロード
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pos));

        // 改行文字と比較
        __m256i cmp = _mm256_cmpeq_epi8(chunk, newline_mask);

        // マッチのビットマスクを取得
        uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(cmp));

        // 各マッチを処理
        while (mask != 0) {
            // 最下位のセットビットの位置を検出
            int bit_pos = ctz(mask);

            const char* newline_pos = pos + bit_pos;
            if (newline_pos + 1 < end) {
                line_starts.push_back(newline_pos + 1);

                if (max_lines > 0 && line_starts.size() > max_lines) {
                    return line_starts;
                }
            }

            // 最下位のセットビットをクリア
            mask &= mask - 1;
        }

        pos += 32;
    }

    // 残りのバイトを処理（32未満）
    while (pos < end) {
        if (*pos == '\n' && pos + 1 < end) {
            line_starts.push_back(pos + 1);

            if (max_lines > 0 && line_starts.size() > max_lines) {
                return line_starts;
            }
        }
        ++pos;
    }

#else
    // AVX2使えない場合
    while (pos < end) {
        if (*pos == '\n' && pos + 1 < end) {
            line_starts.push_back(pos + 1);

            if (max_lines > 0 && line_starts.size() > max_lines) {
                return line_starts;
            }
        }
        ++pos;
    }
#endif

    return line_starts;
}

// AVX2 SIMDを使用して行オフセット（バイト位置）を検出
inline std::vector<size_t> find_line_offsets(const char* data, size_t size, size_t max_lines = 0) {
    std::vector<size_t> offsets;
    offsets.reserve(max_lines > 0 ? max_lines + 1 : size / 50);

    const char* end = data + size;
    const char* pos = data;

    // 最初の行は常にオフセット0から開始
    offsets.push_back(0);

#if HAS_AVX2
    const __m256i newline_mask = _mm256_set1_epi8('\n');

    while (pos + 32 <= end) {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pos));
        __m256i cmp = _mm256_cmpeq_epi8(chunk, newline_mask);
        uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(cmp));

        while (mask != 0) {
            int bit_pos = ctz(mask);
            size_t offset = static_cast<size_t>(pos - data) + bit_pos + 1;

            if (offset < size) {
                offsets.push_back(offset);

                if (max_lines > 0 && offsets.size() > max_lines) {
                    return offsets;
                }
            }

            mask &= mask - 1;
        }

        pos += 32;
    }

    // 残りのバイトを処理
    while (pos < end) {
        if (*pos == '\n') {
            size_t offset = static_cast<size_t>(pos - data) + 1;
            if (offset < size) {
                offsets.push_back(offset);

                if (max_lines > 0 && offsets.size() > max_lines) {
                    return offsets;
                }
            }
        }
        ++pos;
    }

#else
    // フォールバック
    while (pos < end) {
        if (*pos == '\n') {
            size_t offset = static_cast<size_t>(pos - data) + 1;
            if (offset < size) {
                offsets.push_back(offset);

                if (max_lines > 0 && offsets.size() > max_lines) {
                    return offsets;
                }
            }
        }
        ++pos;
    }
#endif

    return offsets;
}

// AVX2 SIMDを使用して複数の5文字整数値をパース
inline void parse_fixed_width_values(const char* data, int count, float* output,
                                     float nodata_value) {
    constexpr int FIELD_WIDTH = 5;

#if HAS_AVX2
    int i = 0;

    for (; i + 4 <= count; i += 4) {
        alignas(32) int results[4];

        for (int j = 0; j < 4; ++j) {
            const char* field = data + (i + j) * FIELD_WIDTH;

            int result = 0;
            int multiplier = 1;
            bool negative = false;

            for (int k = 4; k >= 0; --k) {
                char c = field[k];
                if (c >= '0' && c <= '9') {
                    result += (c - '0') * multiplier;
                    multiplier *= 10;
                } else if (c == '-') {
                    negative = true;
                }
            }

            results[j] = negative ? -result : result;
        }

        // floatに変換してnodataを処理
        for (int j = 0; j < 4; ++j) {
            int val = results[j];
            if (val == -1111 || val == -9999) {
                output[i + j] = nodata_value;
            } else {
                output[i + j] = static_cast<float>(val) * 0.1f;
            }
        }
    }

    // 残りの値を処理
    for (; i < count; ++i) {
        const char* field = data + i * FIELD_WIDTH;

        int result = 0;
        int multiplier = 1;
        bool negative = false;

        for (int k = 4; k >= 0; --k) {
            char c = field[k];
            if (c >= '0' && c <= '9') {
                result += (c - '0') * multiplier;
                multiplier *= 10;
            } else if (c == '-') {
                negative = true;
            }
        }

        int val = negative ? -result : result;
        if (val == -1111 || val == -9999) {
            output[i] = nodata_value;
        } else {
            output[i] = static_cast<float>(val) * 0.1f;
        }
    }

#else

    for (int i = 0; i < count; ++i) {
        const char* field = data + i * FIELD_WIDTH;

        int result = 0;
        int multiplier = 1;
        bool negative = false;

        for (int k = 4; k >= 0; --k) {
            char c = field[k];
            if (c >= '0' && c <= '9') {
                result += (c - '0') * multiplier;
                multiplier *= 10;
            } else if (c == '-') {
                negative = true;
            }
        }

        int val = negative ? -result : result;
        if (val == -1111 || val == -9999) {
            output[i] = nodata_value;
        } else {
            output[i] = static_cast<float>(val) * 0.1f;
        }
    }
#endif
}

// AVX2が利用可能かチェック
inline bool is_avx2_available() {
#if HAS_AVX2
    return true;
#else
    return false;
#endif
}

}  // namespace simd
}  // namespace lem_converter
