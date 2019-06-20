//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
//#include <iostream>
#include <math.h>
//#include <iomanip>

#define X 1
#define Y 4
#define Z 4
#define N 1


static inline int MAX(int a, int b) {
	if (a > b) {
		return a;
	}
	return b;
}

void printMatrixMult(float a[], float b[], float c[], int x, int y, int z){
	//print one line

	//std::cout<<std::setprecision(3);
	//std::cout<<std::fixed;

	int maxi = MAX(z,MAX(x,y));
	for(int line=0;line<maxi;++line){

		//matrix a
		if(line<y){
			printf("[");
			for(int i=0;i<x;++i)
				printf(" %f,",a[line*x+i]);
			printf("]");
		}

		if(line==0)
			printf("*");
		else
			printf(" ");

		//matrix b
		if(line<z){
			printf("[");
			for(int i=0;i<y;++i)
				printf(" %f,",b[line*y+i]);
			printf("]");
		}

		if(line==0)
			printf("=");
		else
			printf(" ");

		//matrix c
		if( line<z){
			printf("[");
			for(int i=0;i<x;++i)
				printf(" %f,",c[line*x+i]);
			printf("]");
		}
		printf("\n");
	}
	printf("\n");
}

//single threaded matrix multiplication
void matrixMult(float xy[], float yz[], float xz[], int q) {
	for (int i = 0; i < X; ++i) {
		for (int j = 0; j < Z; ++j) {
			for (int k = 0; k < Y; ++k) {
				xz[q*X*Z + i*Y + j] += xy[q*X*Y + i*Y + k] * yz[q*Y*Z + k*Z + j];
			}
		}
	}
}

// multi threaded matrix multiplication of a single matrix
void matrixMultSingle(float xy[], float yz[], float xz[], int q ) {
	int index = 0;//blockIdx.x * blockDim.x + threadIdx.x;
	int stride = 1;//blockDim.x * gridDim.x;

	for(int a = index; a < X*Y*Z; a+=stride){
		xz[q*X*Z + a/Y/Z*Y + (a/Z) % Y] += xy[q*X*Y + a/Y/Z*Y + a%Z] * yz[q*Y*Z + a%Z*Z + (a/Z) % Y];
	}
}



// multi threaded multiplication of many matrices
//__global__
void multAll(float matrixListXY[], float matrixListYZ[], float matrixListXZ[]) {

	int index = 0;//blockIdx.x * blockDim.x + threadIdx.x;
	int stride = 1;//blockDim.x * gridDim.x;

	//matrix a size is n*x*y
	//matrix b size is n*y*z


	for (int i = index; i < N; i += stride) {
		matrixMult(matrixListXY, matrixListYZ, matrixListXZ, i );
	}
}

int main(void) {

	//dim3 matrixDim(X, Y, Z);

  //initialization for cuda
	float* matrixListXY, * matrixListYZ;
	//cudaMallocManaged(&matrixListXY, N * X * Y * sizeof(float));
	//cudaMallocManaged(&matrixListYZ, N * Y * Z * sizeof(float));

  //set randomizer
  srand(2);

  //initialize matrices
	float *XY, *YZ, *XZ;

	XY = (float*)malloc(N * X * Y * sizeof(float));
	YZ = (float*)malloc(N * Y * Z * sizeof(float));
	XZ = (float*)malloc(N * X * Z * sizeof(float));

	for (int i = 0; i < N * X * Y; ++i) {
		//XY[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		XY[i] = (float) (i+1);
	}

	for (int i = 0; i < N * Y * Z; ++i) {
		//YZ[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		YZ[i] = (float) (i+1);
	}

	for (int i = 0; i < N * X * Z; ++i) {
		XZ[i] = (float) 0;
	}


  //make events for profiling
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);

  //copy data to GPU mem
	//cudaMemcpy(matrixListXY, XY, N * X * Y * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(matrixListYZ, YZ, N * Y * Z * sizeof(float), cudaMemcpyHostToDevice);

	//cudaEventRecord(start);
  
  //do work
	multAll /*<<< (N+511)/512, 512 >> > */(XY, YZ, XZ);
  
  //copy data back to cpu mem before printintg
  //TO DO
  
  //print if we want
	printMatrixMult(XY,YZ,XZ,X,Y,Z);

  //reset memory
	for (int i = 0; i < N * X * Z; ++i) {
		XZ[i] = (float) 0;
	}

  //copy data to gpu again
  //TO DO
  
  //do work again
	matrixMultSingle(XY,YZ,XZ,0);
  
  //copy data back to cpu again
  //TO DO
  
  //print if we want
	printMatrixMult(XY,YZ,XZ,X,Y,Z);

  //final book keeping
	/*cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0.0f;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Effective Bandwidth (GB/s): %f", ((N * 4 * X * Y )+ (N*4*Y*Z)) / milliseconds / 1e6);
	cudaFree(matrixListXY);
	cudaFree(matrixListYZ);*/
	
	//free(XY);
	//free(YZ);
	//free(XZ);

	return 0;
} 
