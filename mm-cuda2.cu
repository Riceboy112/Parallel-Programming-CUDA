//----------------------------------------------------------
// Matrix Multiplication - CUDA Version 2 to run on GPUs
//---------------------------------------------------------
//  By Gita Alaghband, Lan Vu 
//  Use shared memory with higher access speed
//  Updated in 8/8/2011
//-----------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>    
#include <cuda.h>

using namespace std;
#define TILE 16

//-----------------------------------------------------------------------
//   Get user input for matrix dimension or printing option
//-----------------------------------------------------------------------
bool GetUserInput(int argc, char *argv[],int& n,int& isPrint)
{
	bool isOK = true;

	if(argc < 2) 
	{
		cout << "Arguments:<X> [<Y>]" << endl;
		cout << "X : Matrix size [X x X]" << endl;
		cout << "Y = 1: print the input/output matrix if X < 10" << endl;
		cout << "Y <> 1 or missing: does not print the input/output matrix" << endl;
		isOK = false;
	}
	else 
	{
		//get matrix size
		n = atoi(argv[1]);
		if (n <= 0) 
		{
			cout << "Matrix size must be larger than 0" <<endl;
			isOK = false;
		}

		//is print the input/output matrix
		if (argc >=3)
			isPrint = (atoi(argv[2])==1 && n <=9)?1:0;
		else
			isPrint = 0;
	}
	return isOK;
}

//-----------------------------------------------------------------------
//Initialize the value of matrix x[n x n]
//-----------------------------------------------------------------------
void InitializeMatrix(float** &x,int n,float value)
{
	x = new float*[n];
	x[0] = new float[n*n];
    srand (time(NULL));

	for (int i = 1; i < n; i++)	x[i] = x[i-1] + n;

	for (int i = 0 ; i < n ; i++)
	{
		for (int j = 0 ; j < n ; j++)
		{
            if (value == 1)  // generate input matrices (a and b)
                x[i][j] = (float)((rand()%10)/(float)2);
            else
                x[i][j] = 0;  // initializing resulting matrix
		}
	}
}

//------------------------------------------------------------------
//Delete matrix x[n x n]
//------------------------------------------------------------------
void DeleteMatrix(float **x,int n)
{
	delete[] x[0];
	delete[] x; 
}

//------------------------------------------------------------------
//Print matrix	
//------------------------------------------------------------------
void PrintMatrix(float **x, int n) 
{
	for (int i = 0 ; i < n ; i++)
	{
		cout<< "Row " << (i+1) << ":\t" ;
		for (int j = 0 ; j < n ; j++)
		{
			printf("%.2f\t", x[i][j]);
		}
		cout<<endl ;
	}
}

//-----------------------------------------------------------------------	
//Do Matrix Multiplication - Version 2: use shared memory
//-----------------------------------------------------------------------	
__global__ void MultiplyMatrix_Version2(float* a, float* b, float* c, int n)
{

	__shared__ float A[TILE][TILE];
	__shared__ float B[TILE][TILE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int Row = blockIdx.y*TILE + ty;
	int Col = blockIdx.x*TILE + tx;
	
	float value = 0;

	if (Row < n && Col < n) 
	{
		for (int i = 0; i < n ; i += TILE) 
		{
			// Load the matrices from device memory to shared memory
			// Each thread loads one element of each matrix
			A[ty][tx] = a[ n*Row + (i + tx)]; 
			B[ty][tx] = b[ n*(i + ty) + Col]; 
			// Synchronize to make sure the matrices are loaded
			__syncthreads();	

			// Multiply the two matrices
			// Each thread computes one element of the block sub-matrix
			int m = ((n - i) < TILE)? (n - i): TILE;
			for (int j = 0; j < m; j++)  value += A[ty][j] * B[j][tx];

			// Synchronize to make sure that the preceding computation is done before 
			// loading two new sub-matrices of A and B in the next iteration
			__syncthreads();	
			
		}
		c[Row*n + Col] = value;
	}
}
//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[])
{

    float **a, **b,**c; //host pointers
	float *da, *db, *dc; //device pointers
	int n,isPrint;
	double runtime;
	
	if(GetUserInput(argc,argv,n,isPrint)==false) return 1;

    cout << "cuda 2 - gpu matrix multiplication using shared memory " << endl;
    cout << "matrix size is " << n << endl;
	//Initialize the value of matrix a and vetors x, y
	InitializeMatrix(a,n,1.0);
	InitializeMatrix(b,n,1.0);
	InitializeMatrix(c,n,0.0);

	//Print the input matrices
	if (isPrint==1)
	{
		cout<< "Matrix a[n][n]:" << endl;
		PrintMatrix(a,n); 
		cout<< "Matrix b[n][n]:" << endl;
		PrintMatrix(b,n); 
		
	}
	
	runtime = clock()/(float)CLOCKS_PER_SEC;

	//Declare grid size and block size
	int numblock = n/TILE + ((n%TILE)?1:0);
	dim3 dimGrid(numblock,numblock);	
	dim3 dimBlock(TILE,TILE);	

	//Allocate memory on device
	cudaMalloc((void**)&da, n*n*sizeof(float));
	cudaMalloc((void**)&db, n*n*sizeof(float));
	cudaMalloc((void**)&dc, n*n*sizeof(float));

	//Copy data to the device
	cudaMemcpy(da, a[0], n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b[0], n*n*sizeof(float), cudaMemcpyHostToDevice);

	//Do the matrix multiplication on the device (GPU)
	MultiplyMatrix_Version2<<<dimGrid,dimBlock>>>(da,db,dc,n);
	
    cudaThreadSynchronize();

	//Get results from the device
	cudaMemcpy(c[0],dc, n*n*sizeof(float),cudaMemcpyDeviceToHost);

	runtime = clock() - runtime;

	//The matrix is as below:
	if (isPrint==1)
	{
		cout<< "Matrix c[n][n]:" << endl;
		PrintMatrix(c,n); 
	}

	cout<< "Program runs in " << setiosflags(ios::fixed) << setprecision(8) << (runtime)/float(CLOCKS_PER_SEC) << " seconds\n";

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	DeleteMatrix(a,n);	
	DeleteMatrix(b,n);	
	DeleteMatrix(c,n);	
	
	return 0;
}
