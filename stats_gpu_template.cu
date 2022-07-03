#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>

#define MAXIMUM_VALUE   1000000.0f
#define HANDLE_ERROR( err )  ( HandleError( err, __FILE__, __LINE__ ) )

void HandleError( cudaError_t err, const char *file, int line ) {
  //
  // Handle and report on CUDA errors.
  //
  if ( err != cudaSuccess ) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );

    exit( EXIT_FAILURE );
  }
}

void checkCUDAError( const char *msg, bool exitOnError ) {
  //
  // Check cuda error and print result if appropriate.
  //
  cudaError_t err = cudaGetLastError();

  if( cudaSuccess != err) {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      if (exitOnError) {
        exit(-1);
      }
  }                         
}

void cleanupCuda( void ) {
  //
  // Clean up CUDA resources.
  //

  //
  // Explicitly cleans up all runtime-related resources associated with the 
  // calling host thread.
  //
  HANDLE_ERROR(
         cudaThreadExit()
         );
}

__device__ double device_pow( double x, double y ) {
  //
  // Calculate x^y on the GPU.
  //
  return pow( x, y );
}

//
// PLACE GPU KERNELS HERE - BEGIN
//

//
// PLACE GPU KERNELS HERE - END
//

int main( int argc, char* argv[] ) {
  //
  // Determine min, max, mean, mode and standard deviation of array
  //
  unsigned int array_size, seed, i;
  struct timeval start, end;
  double runtime;

  if( argc < 3 ) {
    printf( "Format: stats_gpu <size of array> <random seed>\n" );
    printf( "Arguments:\n" );
    printf( "  size of array - This is the size of the array to be generated and processed\n" );
    printf( "  random seed   - This integer will be used to seed the random number\n" );        
    printf( "                  generator that will generate the contents of the array\n" );     
    printf( "                  to be processed\n" );   

    exit( 1 );
  }

  //
  // Get the size of the array to process.
  //
  array_size = atoi( argv[1] );

  //
  // Get the seed to be used 
  //
  seed = atoi( argv[2] );

  //
  // Make sure that CUDA resources get cleaned up on exit.
  //
  atexit( cleanupCuda );

  //
  // Record the start time.
  //
  gettimeofday( &start, NULL );

  //
  // Allocate the array to be populated.
  //
  double *array = (double *) malloc( array_size * sizeof( double ) );

  //
  // Seed the random number generator and populate the array with its values.
  //
  srand( seed );
  for( i = 0; i < array_size; i++ )
    array[i] = ( (double) rand() / (double) RAND_MAX ) * MAXIMUM_VALUE;

  //
  // Setup output variables to hold min, max, mean, and standard deviation
  //
  // YOUR CALCULATIONS BELOW SHOULD POPULATE THESE WITH RESULTS
  //
  double min = DBL_MAX;
  double max = 0;
  double sum = 0;
  double mean = 0;
  double stddev = 0;

  //
  // CALCULATE VALUES FOR MIN, MAX, MEAN, and STDDEV - BEGIN
  //

  //
  // CALCULATE VALUES FOR MIN, MAX, MEAN, and STDDEV - END
  //

  //
  // Record the end time.
  //
  gettimeofday( &end, NULL );

  //
  // Calculate the runtime.
  //
  runtime = ( ( end.tv_sec  - start.tv_sec ) * 1000.0 ) + ( ( end.tv_usec - start.tv_usec ) / 1000.0 );    

  //
  // Output discoveries from the array.
  //
  printf( "Statistics for array ( %d, %d ):\n", array_size, seed );
  printf( "    Minimum = %4.6f, Maximum = %4.6f\n", min, max );
  printf( "    Mean = %4.6f, Standard Deviation = %4.6f\n", mean, stddev );
  printf( "Processing Time: %4.4f milliseconds\n", runtime );

  //
  // Free the allocated array.
  //
  free( array );

  return 0;
}
