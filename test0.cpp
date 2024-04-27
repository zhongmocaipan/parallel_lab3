//#include <stdio.h>
//#include <stdlib.h>
//#include <malloc.h>    // 仅在Linux平台上需要包含此头文件
//#include <xmmintrin.h> // SSE指令集头文件
//
//#define N 1000
//
//void Gaussian_Elimination_Unaligned(float* A, float* b) {
//	for (int k = 0; k < N; ++k) {
//		float factor = 1.0f / A[k * N + k];
//		__m128 factor_vec = _mm_set1_ps(factor);
//
//		for (int j = k + 1; j < N; ++j) {
//			__m128 Aj_vec = _mm_loadu_ps(&A[k * N + j]);
//			Aj_vec = _mm_mul_ps(Aj_vec, factor_vec);
//			_mm_storeu_ps(&A[k * N + j], Aj_vec);
//		}
//
//		for (int i = k + 1; i < N; ++i) {
//			float Aik = A[i * N + k];
//			for (int j = k + 1; j < N; ++j) {
//				A[i * N + j] -= Aik * A[k * N + j];
//			}
//			b[i] -= factor * b[k];
//		}
//	}
//}
//
//void Gaussian_Elimination_Aligned(float* A, float* b) {
//	for (int k = 0; k < N; ++k) {
//		float factor = 1.0f / A[k * N + k];
//		__m128 factor_vec = _mm_set1_ps(factor);
//
//		// 将 k 到对齐边界的部分串行处理
//		int aligned_end = (N - k) & (~3);
//		for (int j = k + 1; j < k + 1 + aligned_end; j += 4) {
//			__m128 Aj_vec = _mm_load_ps(&A[k * N + j]);
//			Aj_vec = _mm_mul_ps(Aj_vec, factor_vec);
//			_mm_store_ps(&A[k * N + j], Aj_vec);
//		}
//
//		// 对齐边界之后的部分使用 SIMD 计算
//		for (int j = k + 1 + aligned_end; j < N; ++j) {
//			A[k * N + j] *= factor;
//		}
//
//		for (int i = k + 1; i < N; ++i) {
//			float Aik = A[i * N + k];
//			for (int j = k + 1; j < N; ++j) {
//				A[i * N + j] -= Aik * A[k * N + j];
//			}
//			b[i] -= factor * b[k];
//		}
//	}
//}
//
//int main() {
//	float* A = (float*)_aligned_malloc(N * N * sizeof(float), 16); // 使用_aligned_malloc分配对齐的内存
//	float* b = (float*)_aligned_malloc(N * sizeof(float), 16);     // 使用_aligned_malloc分配对齐的内存
//
//	if (A == NULL || b == NULL) {
//		printf("Memory allocation failed\n");
//		return 1;
//	}
//
//	// 初始化 A 和 b，这里简单起见，直接将所有元素置为随机数
//	for (int i = 0; i < N * N; ++i) {
//		A[i] = (float)rand() / RAND_MAX;
//	}
//	for (int i = 0; i < N; ++i) {
//		b[i] = (float)rand() / RAND_MAX;
//	}
//
//	// 测试不对齐算法
//	Gaussian_Elimination_Unaligned(A, b);
//
//	// 测试对齐算法
//	Gaussian_Elimination_Aligned(A, b);
//
//	_aligned_free(A); // 使用_aligned_free释放内存
//	_aligned_free(b); // 使用_aligned_free释放内存
//
//	return 0;
//}


#include <iostream>
#include <immintrin.h>

#define N 4

void gaussian_elimination_x86(float* A, float* b) {
	for (int i = 0; i < N; i++) {
		float scale = 1.0f / A[i * N + i];
		__m128 scale_vec = _mm_set1_ps(scale);
		for (int j = 0; j < N; j += 4) {
			__m128 A_row = _mm_loadu_ps(&A[j * N + i]);
			__m128 scaled_row = _mm_mul_ps(A_row, scale_vec);
			_mm_storeu_ps(&A[j * N + i], scaled_row);
		}
		b[i] *= scale;
		for (int j = 0; j < N; j++) {
			if (i != j) {
				float ratio = A[j * N + i];
				__m128 ratio_vec = _mm_set1_ps(ratio);
				for (int k = 0; k < N; k += 4) {
					__m128 A_row_i = _mm_loadu_ps(&A[i * N + k]);
					__m128 A_row_j = _mm_loadu_ps(&A[j * N + k]);
					__m128 scaled_row_i = _mm_mul_ps(A_row_i, ratio_vec);
					__m128 sub_row = _mm_sub_ps(A_row_j, scaled_row_i);
					_mm_storeu_ps(&A[j * N + k], sub_row);
				}
				b[j] -= ratio * b[i];
			}
		}
	}
}

int main() {
	float A[N][N] = { {4, 1, -2, 2},
					 {1, 2, 0, 1},
					 {0, 3, -3, 1},
					 {5, 1, 2, 3} };
	float b[N] = { 7, 8, 9, 10 };

	gaussian_elimination_x86((float*)A, b);

	std::cout << "Solution vector x:" << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << b[i] << " ";
	}
	std::cout << std::endl;

	return 0;
}
