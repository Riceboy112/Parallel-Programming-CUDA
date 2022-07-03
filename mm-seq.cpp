//-----------------------------------------------------------------------
// Matrix Multiplication - Sequential version to run on single CPU core only
//-----------------------------------------------------------------------
//  Parallel and Distributed System (PDS) Lab
//  Updated in 8/8/2011
//-----------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <time.h>
#include <cstdlib>
using namespace std;
//-----------------------------------------------------------------------
//   Get user input for matrix dimension or printing option
//-----------------------------------------------------------------------

typedef float** twoDPtr; 

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
		if (n <=0) 
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
float** InitializeMatrix(int n, float value)
{

	// allocate square 2d matrix
	float **x = new float*[n];
	for(int i = 0 ; i < n ; i++)
		x[i] = new float[n] ;


	// assign random values
    srand (time(NULL));
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

	return x ;
}
//------------------------------------------------------------------
//Delete matrix x[n x n]
//------------------------------------------------------------------
void DeleteMatrix(float **x,int n)
{

	for(int i = 0; i < n ; i++)
		delete[] x[i];
	
}
//------------------------------------------------------------------
//Print matrix	
//------------------------------------------------------------------
void PrintMatrix(float** x, int n) 
{

	for (int i = 0 ; i < n ; i++)
	{
		cout<< "Row " << (i+1) << ":\t" ;
		for (int j = 0 ; j < n ; j++)
		{
			cout << setiosflags(ios::fixed) << setprecision(2) << x[i][j] << " ";
		}
		cout << endl ;
	}
}
//------------------------------------------------------------------
//Do Matrix Multiplication 
//------------------------------------------------------------------
void MultiplyMatrix(float** a, float** b, float** c, int n)
{
	for (int i = 0 ; i < n ; i++)
		for (int j = 0 ; j < n ; j++)
			for (int k = 0 ; k < n ; k++)
				c[i][j] += a[i][k]*b[k][j];
}
//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[])
{
	int	n,isPrint;
	double runtime;

	if (GetUserInput(argc,argv,n,isPrint)==false) return 1;

    cout << "Starting sequential matrix multiplication" << endl;
    cout << "matrix size = " << n << "x " << n << endl;

	//Initialize the value of matrix a, b, c
	float **a = InitializeMatrix(n, 1.0);
	float **b = InitializeMatrix(n, 1.0);
	float **c = InitializeMatrix(n, 0.0);

	//Print the input matrices
	if (isPrint==1)
	{
		cout<< "Matrix a[n][n]:" << endl;
		PrintMatrix(a,n); 
		cout<< "Matrix b[n][n]:" << endl;
		PrintMatrix(b,n); 
	}

	runtime = clock()/(double)CLOCKS_PER_SEC;

	MultiplyMatrix(a,b,c,n);

	runtime = (clock()/(double)CLOCKS_PER_SEC ) - runtime;

	//Print the output matrix
	if (isPrint==1)
	{
		cout<< "Output matrix:" << endl;
		PrintMatrix(c,n); 
	}
	cout<< "Program runs in " << setiosflags(ios::fixed) << setprecision(8) << runtime << " seconds\n";
	
	DeleteMatrix(a,n);	
	DeleteMatrix(b,n);	
	DeleteMatrix(c,n);	

	return 0;
}
