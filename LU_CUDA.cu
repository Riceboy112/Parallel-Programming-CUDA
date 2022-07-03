//----------------------------------------------------------
// Matrix Multiplication - CUDA Version 1 to run on GPUs
//---------------------------------------------------------
//  By Parallel and Distributed System (PDS) Lab
//  Updated in 08/2020
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
//Do Matrix Multiplication - Version 1: not use shared memory 
//-----------------------------------------------------------------------
__global__ void MultiplyMatrix_Version1(float* matrix, float* lower, float* upper, int size)
{
	int Row = blockIdx.x*TILE + threadIdx.x;
	int Col = blockIdx.y*TILE + threadIdx.y;
	/*
	if (Row < n  && Col < n)
	{
		float value = 0;
		
		for (int i = 0; i < n; i++) value += a[Row*n + i] * b[i*n + Col];
			
		c[Row*n + Col] = value;
	}		*/


	if (Row < n && Col < n)
	{
	for (int i = 0; i < size; i++)
    {
        // Upper Triangular
        for (int k = i; k < size; k++)
        {
            // Summation of L(i, j) * U(j, k)
            int sum = 0;
            for (int j = 0; j < i; j++)
                sum += (lower[i][j] * upper[j][k]);
                
            float value = 0;
            value = matrix[i][k] - sum;
            // Evaluating U(i, k)
            upper[Row*n][k] = value;
        }
        
        // Lower Triangular
        for (int k = i; k < size; k++)
        {
            if (i == k)
                lower[i][i] = 1; // Diagonal as 1
            else
            {
                // Summation of L(k, j) * U(j, i)
                int sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (lower[k][j] * upper[j][i]);
                
                // Evaluating L(k, i)
                lower[k][i] = (matrix[k][i] - sum) / upper[i][i];
            }
        }
    }
    }
	
	
	
	
	
	
	
}

//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[])
{


    float **a, **l,**u; //host pointers
	float *da, *dl, *du; //device pointers
	int n,isPrint;
	double runtime;
	
	if(GetUserInput(argc,argv,n,isPrint)==false) return 1;
    cout << "Cuda 1 - gpu matrix multiplication with NO shared memory " << endl;
    cout << "matrix size is " << n << endl;
	//Initialize the value of matrix a and vetors x, y
	InitializeMatrix(a,n,1.0);
	InitializeMatrix(l,n,0.0);
	InitializeMatrix(u,n,0.0);

	//Print the input matrices
	if (isPrint==1)
	{
		cout<< "Matrix a[n][n]:" << endl;
		PrintMatrix(a,n); 
		//cout<< "Matrix b[n][n]:" << endl;
		//PrintMatrix(b,n); 
	}
	
	runtime = clock()/(float)CLOCKS_PER_SEC;

	//Declare grid size and block size
	int numblock = n/TILE + ((n%TILE)?1:0);
	dim3 dimGrid(numblock,numblock);	
	dim3 dimBlock(TILE,TILE);	

	//Allocate memory on device
	cudaMalloc((void**)&da, n*n*sizeof(float));
	cudaMalloc((void**)&dl, n*n*sizeof(float));
	cudaMalloc((void**)&du, n*n*sizeof(float));

	//Copy data to the device
	cudaMemcpy(da, a[0], n*n*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(db, b[0], n*n*sizeof(float), cudaMemcpyHostToDevice);

	//Do the matrix multiplication on the device (GPU)
	MultiplyMatrix_Version1<<<dimGrid,dimBlock>>>(da,dl,du,n);
	
    cudaThreadSynchronize();

	//Get results from the device
	cudaMemcpy(c[0],dc, n*n*sizeof(float),cudaMemcpyDeviceToHost);

	runtime = clock() - runtime;

	//Print the output matrix
	if (isPrint==1)
	{
		cout<< "Matrix c[n][n]:" << endl;
		PrintMatrix(c,n); 
	}

	cout<< "Program runs in " << setiosflags(ios::fixed) << setprecision(2) << (runtime)/float(CLOCKS_PER_SEC) << " seconds\n";

	cudaFree(da);
	cudaFree(dl);
	cudaFree(du);

	DeleteMatrix(a,n);	
	DeleteMatrix(l,n);	
	DeleteMatrix(u,n);	
	
	return 0;
}