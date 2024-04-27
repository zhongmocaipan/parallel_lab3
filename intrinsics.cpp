#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define N 1000

void generate_random_matrix(float* A, float* b) {
	// 使用不同的种子来生成随机数
	srand((unsigned int)time(NULL));
	for (int i = 0; i < N; i++) {
		float row_sum = 0.0f; // 用于归一化处理
		for (int j = 0; j < N; j++) {
			A[i * N + j] = (float)rand() / RAND_MAX;
			row_sum += A[i * N + j];
		}
		// 归一化处理
		for (int j = 0; j < N; j++) {
			A[i * N + j] /= row_sum;
		}
		b[i] = (float)rand() / RAND_MAX;
	}
}

void print_matrix(float* A) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%.2f ", A[i * N + j]);
		}
		printf("\n");
	}
}

void print_vector(float* b) {
	for (int i = 0; i < N; i++) {
		printf("%.2f ", b[i]);
	}
	printf("\n");
}

void gaussian_elimination(float* A, float* b) {
	for (int i = 0; i < N; i++) {
		float scale = 1.0f / A[i * N + i];
		__m128 scale_vec = _mm_set1_ps(scale);
		for (int j = 0; j < N; j += 4) {
			if (A && b) {
				__m128 A_row = _mm_load_ps(&A[j * N + i]);
				__m128 scaled_row = _mm_mul_ps(A_row, scale_vec);
				_mm_store_ps(&A[j * N + i], scaled_row);
			}
		}
		if (b && A[i * N + i] != 0.0f) { // 避免除以零的情况
			b[i] *= scale;
		}
		for (int j = 0; j < N; j++) {
			if (i != j && A && b && A[i * N + i] != 0.0f) {
				float ratio = A[j * N + i];
				__m128 ratio_vec = _mm_set1_ps(ratio);
				for (int k = 0; k < N; k += 4) {
					__m128 A_row_i = _mm_load_ps(&A[i * N + k]);
					__m128 A_row_j = _mm_load_ps(&A[j * N + k]);
					__m128 scaled_row_i = _mm_mul_ps(A_row_i, ratio_vec);
					__m128 sub_row = _mm_sub_ps(A_row_j, scaled_row_i);
					_mm_store_ps(&A[j * N + k], sub_row);
				}
				b[j] -= ratio * b[i];
			}
		}
	}
}

int main() {
	float A[N][N];
	float b[N];
	float* A_flat = (float*)A;
	generate_random_matrix(A_flat, b);

	printf("Original matrix A:\n");
	print_matrix(A_flat);
	printf("Original vector b:\n");
	print_vector(b);

	gaussian_elimination(A_flat, b);

	printf("Solution vector x:\n");
	print_vector(b);

	return 0;
}
