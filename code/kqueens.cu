#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define NUM_BLOCKS 16
#define NUM_THREADS 16
#define k 80
#define MAX_ITER 4000

using namespace std;

__device__ int checkDiagonals(int q,int i, int* S)
// Returns 1 if no queen in diagonal, else 0
{
	int I = blockIdx.x*NUM_THREADS*k + threadIdx.x*k;
	int j = 1;
	for (j; j<=i; j++){
		if (S[I+i-j] == q-j | S[I+i-j] == q+j){
			return 0;
		} 		
	}	
	return 1;
}

__device__ int sum(int row[], int len)
// Returns sum of an array 
{
	int i = 0;
	int s = 0;
	for (i; i<len; i++){
		s += row[i];
	}
	return s;
}

__global__ void setup_kernel (curandState * state, unsigned long seed)
// Create states to generate random numbers
{
	int id = blockIdx.x*NUM_BLOCKS + threadIdx.x;
	curand_init ( seed, id, 0, &state[id] );
}

__global__ void kernel(int* S, curandState* globalState)
// Kernel to solve puzzle
{
	// Index for thread to store solution
	int I = blockIdx.x*NUM_THREADS*k + threadIdx.x*k;
	int ind = blockIdx.x*NUM_BLOCKS + threadIdx.x;
	int D[k];				// Rows where queens is placed. 1 = row taken
	int N[k][k];				// Positions tried at column i
	
	int i = 0;				
	int j = 0;

	int q; 
	
	// Initialize variables
	for (i; i<k; i++){
		S[I+i] = -1;
		D[i] = 0;
		for (j;j<k;j++){
			N[i][j] = 0;
		}
		j = 0;
	}

	// Set start column and iter counter
	i = 0;
	int iter = 0;

	// Get local state to generate numbers 
	curandState localState = globalState[ind];
	
	while (iter < MAX_ITER){
	
		// Generate random number
		q = curand_uniform( &localState ) * k;
		
		if (D[q] == 0 & N[i][q] == 0){ 		// Row clear and not tried before 
			N[i][q] = 1;				// Set position as tried
			if (checkDiagonals(q,i,S)==1){	// If no attacking queens in diagonal
				S[I+i] = q;			// Add queen to solution
				D[q] = 1;			// Set row as taken
				i++;				// Increment interation counter
				if (i==k){			// Finished!
					break;
				}
			}
		}
		if (sum(N[i],k) + sum(D,k) == k){ 		// All positions tried
			D[S[I+i-1]] = 0;					// Free domain
			S[I+i-1] = -1;						// Remove queen from solution
			j = 0;	
			for (j;j<k;j++){		// Reset positions tried for column
				N[i][j] = 0;
			}		
			i--;				// Backtrack to prevoius column
		}
		iter++;
	}
}

int main() 
{
	// Initialize states variable and allocate memory
	curandState* devStates;
	cudaMalloc ( &devStates, k*sizeof( curandState ) );

	// Initialze seeds
	setup_kernel <<< NUM_BLOCKS, NUM_THREADS>>> ( devStates,unsigned(time(NULL)) );

	// Initialize array to store solution
	int solution_host[k*NUM_BLOCKS*NUM_THREADS];
	int* solution_dev;

	// Allocate memory on device
	cudaMalloc((void**) &solution_dev, sizeof(int)*k*NUM_BLOCKS*NUM_THREADS);

	// Start clock
	clock_t begin = clock();
	// Launch kernel on device
	kernel<<<NUM_BLOCKS,NUM_THREADS>>> (solution_dev, devStates);
	// Copy solution from device to host
	cudaMemcpy(solution_host, solution_dev, sizeof(int)*k*NUM_BLOCKS*NUM_THREADS, cudaMemcpyDeviceToHost);
	// End clock
	clock_t end = clock();
	
	double elapsed_sec = double(end - begin)/(CLOCKS_PER_SEC/1000);

	// Print time used
	cout << elapsed_sec << endl;

	// Count solutions found (not -1 in last position)
	int solution_count = 0;
	for (int l=0;l<NUM_BLOCKS*NUM_THREADS;l++){
		if (solution_host[l*k+k-1]!=-1){
			solution_count++;
		}
	}
	// Print solutions found
	cout << solution_count << endl;

	// Free memory on device
	cudaFree(devStates);
	cudaFree(solution_dev);

	return 0;
}
