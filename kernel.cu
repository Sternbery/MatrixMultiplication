
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <iomanip>

#define X 1
#define Y 4
#define Z 4
#define N ((1<<20))

__global__
void multAll(float matrixListXY[], float matrixListYZ[]);
__device__
void matrixMult(float a[], float b[], int q);

//__device__
//void printMatrixMult(float a[], float b[], float c[]);



__global__
void multAll(float matrixListXY[], float matrixListYZ[]) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	//matrix a size is n*x*y
	//matrix b size is n*y*z
	

	for (int i = index; i < N; i += stride) {
		matrixMult(matrixListXY, matrixListYZ, i);
	}
}

__device__
void matrixMult(float a[], float b[], int q) {


	float* mult = (float*)malloc(X * Z*sizeof(float));
	
	for (int i = 0; i < X; ++i) {
		for (int j = 0; j < Z; ++j)
		{
			mult[j * X + i] = 0;
		}
	}

	for (int i = 0; i < X; ++i) {
		for (int j = 0; j < Z; ++j) {
			for (int k = 0; k < Y; ++k) {
				//this may not dereference correctly
				//std::cout<<"a is of type: "<<typeid(q*x*y+i*y+j).name()<<std::endl;
				//std::cout << a[q * X * Y + k * Y + i] << " * " << b[q * Y * Z + k * Z + j] << std::endl;
				mult[j * X + i] += a[q * X * Y + i * Y + k] * b[q * Y * Z + k * Z + j];
			}
		}
	}


	//if (true)
	//	return;

	//float* ap, * bp;
	//cudaMallocManaged(&ap, X * Y * sizeof(float));
	//cudaMallocManaged(&bp, Y * Z * sizeof(float));

	//for (int i = 0; i < X; ++i) {
	//	for (int j = 0; j < Y; ++j)
	//	{
	//		ap[j * X + i] = a[q * X * Y + i * Y + j];

	//	}
	//}

	//for (int i = 0; i < Y; ++i) {
	//	for (int j = 0; j < Z; ++j)
	//	{
	//		bp[j * Y + i] = b[q * Y * Z + i * Y + j];
	//	}
	//}

	//printMatrixMult(ap, bp, mult);

	free(mult);
	//cudaFree(ap);
	//cudaFree(bp);
}

static inline int MAX(int a, int b) {
	if (a > b) {
		return a;
	}
	return b;
}
//__device__
//void printmatrixmult(float a[], float b[], float c[]) {
//	print one line
//
//	std::cout << std::setprecision(3);
//	std::cout << std::fixed;
//
//	int maxi = max(z, max(x, y));
//	for (int line = 0; line < maxi; ++line) {
//
//		matrix a
//		if (line < y) {
//			std::cout << "[";
//			for (int i = 0; i < x; ++i)
//				std::cout << " " << a[line * x + i] << ",";
//			std::cout << "]";
//		}
//
//		if (line == 0)
//			std::cout << "*";
//		else
//			std::cout << " ";
//
//		matrix b
//		if (line < z) {
//			std::cout << "[";
//			for (int i = 0; i < y; ++i)
//				std::cout << " " << b[line * y + i] << ",";
//			std::cout << "]";
//		}
//
//		if (line == 0)
//			std::cout << "=";
//		else
//			std::cout << " ";
//
//		matrix c
//		if (line < z) {
//			std::cout << "[";
//			for (int i = 0; i < x; ++i)
//				std::cout << " " << c[line * x + i] << ",";
//			std::cout << "]";
//		}
//		std::cout << "\n";
//	}
//	std::cout << "\n";
//}

int main(void) {


	dim3 matrixDim(X, Y, Z);

	srand(2);

	float* matrixListXY, * matrixListYZ, *XY, *YZ;

	XY = (float*)malloc(N * X * Y * sizeof(float));
	YZ = (float*)malloc(N * Y * Z * sizeof(float));

	cudaMallocManaged(&matrixListXY, N * X * Y * sizeof(float));
	cudaMallocManaged(&matrixListYZ, N * Y * Z * sizeof(float));



	for (int i = 0; i < N * X * Y; ++i) {
		XY[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}

	for (int i = 0; i < N * Y * Z; ++i) {
		YZ[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(matrixListXY, XY, N * X * Y * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(matrixListYZ, YZ, N * Y * Z * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	multAll <<< (N+511)/512, 512 >> > (matrixListXY, matrixListYZ);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0.0f;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Effective Bandwidth (GB/s): %f", ((N * 4 * X * Y )+ (N*4*Y*Z)) / milliseconds / 1e6);
	cudaFree(matrixListXY);
	cudaFree(matrixListYZ);

	free(XY);
	free(YZ);
	return 0;
}

 