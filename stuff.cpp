#include <stdlib.h> //for random number generator
#include <stdio.h>
#include <iostream>
#include <math.h>
//#include <typeinfo>
#include <iomanip>


void printMatrixMult(float a[], float b[], float c[], int x, int y, int z){
	//print one line
	
	std::cout<<std::setprecision(3);
	std::cout<<std::fixed;
	
	int maxi = std::max(z,std::max(x,y));
	for(int line=0;line<maxi;++line){
		
		//matrix a
		if(line<y){
			std::cout<<"[";
			for(int i=0;i<x;++i)
				std::cout<<" "<<a[line*x+i]<<",";
			std::cout<<"]";
		}
		
		if(line==0)
			std::cout<<"*";
		else
			std::cout<<" ";
		
		//matrix b
		if(line<z){
			std::cout<<"[";
			for(int i=0;i<y;++i)
				std::cout<<" "<<b[line*y+i]<<",";
			std::cout<<"]";
		}
		
		if(line==0)
			std::cout<<"=";
		else
			std::cout<<" ";
			
		//matrix c
		if( line<z){
			std::cout<<"[";
			for(int i=0;i<x;++i)
				std::cout<<" "<<b[line*x+i]<<",";
			std::cout<<"]";
		}
		std::cout<<"\n";
	}
	std::cout<<"\n";
}

void matrixMult(float a[], float b[], int q,int x, int y, int z){

	float mult[x*z];
	for(int i = 0; i < x; ++i)
		for(int j = 0; j < z; ++j)
		{
			mult[i*x+j]=0;
		}
	
	for(int i=0;i<x;++i){
		for(int j=0;j<z;++j){
			for(int k=0;k<y;++k){
				//this may not dereference correctly
				//std::cout<<"a is of type: "<<typeid(q*x*y+i*y+j).name()<<std::endl;
				mult[i*x+j] += a[q*x*y+i*y+j] * b[q*y*z+k*z+j];
			}
		}
	}

	
	if(false)
		return;
	
	float ap[x*y];
	float bp[y*z];
	
	for(int i = 0; i < x; ++i)
		for(int j = 0; j < y; ++j)
		{
			ap[i*x+j]=a[q*x*y+i*y+j];
			
		}
	for(int i = 0; i < y; ++i)
		for(int j = 0; j < z; ++j)
		{
			bp[i*x+j]=b[q*x*y+i*y+j];
		}
	printMatrixMult(ap,bp,mult,x,y,z);
}

//__global__
void multAll(float a[], float b[], int n, int x, int y, int z){

	int index = 0;//blockIdx.x * blockDim.x + threadIdx.x;
	int stride = 1;//blockDim.x * gridDim.x;

	//matrix a size is n*x*y
	//matrix b size is n*y*z
	//
	
	for(int i=index;i<n;i+=stride){
		matrixMult(a,b,i,x,y,z);
	}
}

int main(){

	float testFloat;
	int fsize = sizeof(testFloat);

	srand(2);
	
	int n = 5;
	int x = 1;
	int y = 4;
	int z = 4;
	
	int asize = x*y*fsize; 
	int bsize = y*z*fsize;
	
	float a[n*x*y];
	for(int i=0; i<n*x*y; ++i){
	    a[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}
	
	float b[n*y*z];
	for(int i=0; i<n*y*z; ++i){
	    b[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}
	
	
	multAll/*<<<numBlocks, blockSize>>>*/(a,b,n,x,y,z);

	printf("float size: %d", fsize);

}

//https://devblogs.nvidia.com/even-easier-introduction-cuda/