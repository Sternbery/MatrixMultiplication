
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <iomanip>

#define X 4
#define Y 4
#define Z 1
#define N 5

void multAll(float matrixListXY[], float matrixListYZ[]);
void matrixMult(float a[], float b[], int q);
void printMatrixMult(float a[], float b[], float c[]);

int main(void) {


	dim3 matrixDim(X, Y, Z);

	srand(2);

	float matrixListXY[N * X * Y];
	float matrixListYZ[N * Y * Z];


	for (int i = 0; i < N * X * Y; ++i) {
		matrixListXY[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}

	for (int i = 0; i < N * Y * Z; ++i) {
		matrixListYZ[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}

	multAll(matrixListXY, matrixListYZ);


	return 0;
}


void multAll(float matrixListXY[], float matrixListYZ[]) {

	int index = 0;//blockIdx.x * blockDim.x + threadIdx.x;
	int stride = 1;//blockDim.x * gridDim.x;

	//matrix a size is n*x*y
	//matrix b size is n*y*z
	

	for (int i = index; i < N; i += stride) {
		matrixMult(matrixListXY, matrixListYZ, i);
	}
}

void matrixMult(float a[], float b[], int q) {


	float mult[X * Z];
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
				mult[j * X + i] += a[q * X * Y + k * Y + i] * b[q * Y * Z + k * Z + j];
			}
			std::cout << std::endl;
		}
	}


	if (false)
		return;

	float ap[X * Y];
	float bp[Y * Z];

	for (int i = 0; i < X; ++i) {
		for (int j = 0; j < Y; ++j)
		{
			ap[j * X + i] = a[q * X * Y + i * Y + j];

		}
	}

	for (int i = 0; i < Y; ++i) {
		for (int j = 0; j < Z; ++j)
		{
			bp[j * Y + i] = b[q * Y * Z + i * Z + j];
		}
	}

	printMatrixMult(ap, bp, mult);
}

static inline int MAX(int a, int b) {
	if (a > b) {
		return a;
	}
	return b;
}

void printMatrixMult(float a[], float b[], float c[]) {
	//print one line

	std::cout << std::setprecision(3);
	std::cout << std::fixed;

	int maxi = MAX(Z, MAX(X, Y));
	for (int line = 0; line < maxi; ++line) {

		//matrix a
		if (line < Y) {
			std::cout << "[";
			for (int i = 0; i < X; ++i)
				std::cout << " " << a[line * X + i] << ",";
			std::cout << "]";
		}

		if (line == 0)
			std::cout << "*";
		else
			std::cout << " ";

		//matrix b
		if (line < Z) {
			std::cout << "[";
			for (int i = 0; i < Y; ++i)
				std::cout << " " << b[line * Y + i] << ",";
			std::cout << "]";
		}

		if (line == 0)
			std::cout << "=";
		else
			std::cout << " ";

		//matrix c
		if (line < Z) {
			std::cout << "[";
			for (int i = 0; i < X; ++i)
				std::cout << " " << c[line * X + i] << ",";
			std::cout << "]";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

