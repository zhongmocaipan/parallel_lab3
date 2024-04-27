#include <stdio.h>
#include <immintrin.h>

#define N 16

void gaussian_elimination(float* A, float* b) {
	__m128 scale_vec = _mm_set1_ps(0.0f);
	for (int i = 0; i < N; i += 4) {
		__m128 A_row = _mm_loadu_ps(&A[i]);
		__m128 scaled_row = _mm_mul_ps(A_row, scale_vec);
		_mm_storeu_ps(&A[i], scaled_row);
	}
	for (int i = 0; i < N; i++) {
		b[i] *= 0.0f;
	}
}

int main() {
	float A[N] = { 4.0f, 1.0f, -2.0f, 2.0f, 1.0f, 2.0f, 0.0f, 1.0f,
				  0.0f, 3.0f, -3.0f, 1.0f, 5.0f, 1.0f, 2.0f, 3.0f };
	float b[N] = { 7.0f, 8.0f, 9.0f, 10.0f, 7.0f, 8.0f, 9.0f, 10.0f,
				  7.0f, 8.0f, 9.0f, 10.0f, 7.0f, 8.0f, 9.0f, 10.0f };

	printf("Original matrix A:\n");
	for (int i = 0; i < N; i++) {
		printf("%.2f ", A[i]);
	}
	printf("\n");

	printf("Original vector b:\n");
	for (int i = 0; i < N; i++) {
		printf("%.2f ", b[i]);
	}
	printf("\n");

	gaussian_elimination(A, b);

	printf("Modified matrix A:\n");
	for (int i = 0; i < N; i++) {
		printf("%.2f ", A[i]);
	}
	printf("\n");

	printf("Modified vector b:\n");
	for (int i = 0; i < N; i++) {
		printf("%.2f ", b[i]);
	}
	printf("\n");

	return 0;
}
