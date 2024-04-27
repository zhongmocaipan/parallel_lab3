#include <stdio.h>
#include <stdlib.h>
#include <malloc.h> // 包含 _aligned_malloc 和 _aligned_free 函数的头文件
#include <immintrin.h>

#define N 100

void* aligned_alloc(size_t alignment, size_t size) {
	return _aligned_malloc(size, alignment); // 使用 _aligned_malloc 分配对齐的内存
}

void free_aligned(void* ptr) {
	_aligned_free(ptr); // 使用 _aligned_free 释放对齐的内存
}

void gaussian_elimination(float* A, float* b) {
	__m128 scale_vec = _mm_set1_ps(0.0f);
	for (int i = 0; i < N; i += 4) {
		// Load aligned data using _mm_load_ps
		__m128 A_row = _mm_load_ps(&A[i]);
		__m128 scaled_row = _mm_mul_ps(A_row, scale_vec);
		_mm_store_ps(&A[i], scaled_row);
	}
	for (int i = 0; i < N; i++) {
		b[i] *= 0.0f;
	}
}

int main() {
	// Allocate aligned memory for A and b
	float* A = (float*)aligned_alloc(16, N * sizeof(float));
	float* b = (float*)aligned_alloc(16, N * sizeof(float));

	// Initialize A and b with some data

	// Perform Gaussian elimination
	gaussian_elimination(A, b);

	// Free allocated memory
	free_aligned(A);
	free_aligned(b);

	return 0;
}
