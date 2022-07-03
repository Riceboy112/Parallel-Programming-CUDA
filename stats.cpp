//----------------------------------------------------------
// stats.cpp- CPU sequential version to calcualte standard variance.
//---------------------------------------------------------
//  By PDS Lab
//  Updated in 8/2020
//-----------------------------------------------------------


#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cfloat>
#include <sys/time.h>

using namespace std; 

void seq_var(const float array[], int array_size, float &min, float &max, float &var){
  /*
  Sequential version of standard variance
  */
  float avg = 0; 
  for(int i = 0; i < array_size; i++ ) {
	avg += array[i];
	if (min > array[i]) min = array[i];
	if (max < array[i]) max = array[i];
  }
  avg = avg/array_size;
  var = 0;
  for(int i = 0; i < array_size; ++i){
	var += (array[i]-avg)*(array[i]-avg);
  }
  var /= array_size;
}


int main( int argc, char* argv[] ) {
  /*
  main program 
  */

  if(argc < 3) {
	cout << "Format: stats_s <size of array> <random seed>" << endl  ;
	cout << "Arguments:" << endl;
	cout << "  size of array - This is the size of the array to be generated and processed\n"  << endl ;
	cout << "  random seed   - This integer will be used to seed the random number\n"  << endl ;        
	cout << "                  generator that will generate the contents of the array\n"  << endl ;     
	cout << "                  to be processed\n"  << endl ;   
	exit(1);
  }

  //seed for randomization
  unsigned int seed = atoi( argv[2] );
  srand( seed );

  // start timer
  struct timeval start, end;
  gettimeofday( &start, NULL ); 
  
  // allocate array 
  unsigned int array_size = atoi(argv[1]);
  float *array = new float[array_size] ;
  float var = 0, min = 100.0, max = -100.0;
  for(int i = 0; i < array_size; i++ ){
	array[i] =  random()%10; 
	//cout << array[i] << " " ;
	}

 
  //calculate min, max, variance
  seq_var(array, array_size, min, max, var);

  //calculate run time
  gettimeofday( &end, NULL );
  float runtime = ( ( end.tv_sec  - start.tv_sec ) * 1000.0 ) + ( ( end.tv_usec - start.tv_usec ) / 1000.0 );    


  // print out
  cout << "Statistics for array size:" << array_size << " seed:"<<  seed << endl ; 
  cout << "    Variance: " << var << endl ;
  cout << "    Min: " << min << endl ;
  cout << "    Max: " << max << endl ;
  cout << "Processing Time:" << runtime << "(ms)" << endl ;

  delete array ;
  return 0;
}
