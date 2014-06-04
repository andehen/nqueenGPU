#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define k 10 // set problem size
#define NUM_BLOCKS 4
#define NUM_THREADS 4


using namespace std;

__device__ float generate(curandState* globalState, int ind2)
// Function to generate random number in thread
{
	int ind = threadIdx.x*blockIdx.x;
	curandState localState = globalState[ind];
	float rnd = curand_uniform( &localState );
	globalState[ind] = localState;
	return rnd;
}

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

__device__ int sum(int* row, int len)
// Returns sum of an array 
{
	int I = blockIdx.x*NUM_THREADS*k + threadIdx.x*k;
	int i = 0;
	int s = 0;
	for (i; i<len; i++){
		s += row[I+i];
	}
	return s;
}

//__device__ int sumN(int* N, int i, int len)
//{
//	int I = blockIdx.x*NUM_THREADS*k + threadIdx.x*k;
//	int s = 0;
//	for (int j=0; j<len; j++){
//		s += N[I+i][j];
//	}
//	return s;
//}

__global__ void setup_kernel (curandState * state, unsigned long seed)
{
	int id = threadIdx.x;
	curand_init ( seed, id, 0, &state[id] );
}

__global__ void kernel(int* S, curandState* globalState)
{

	int I = blockIdx.x*NUM_THREADS*k + threadIdx.x*k;
	// Initialize varaibles
	//int S[k]; 				// Holds current solution
	//__shared__ int S[NUM_BLOCKS*NUM_THREADS*k];
	int D[NUM_BLOCKS*NUM_THREADS*k];				// Rows where queens is placed
	int N[NUM_BLOCKS*NUM_THREADS*k][k];				// Positions tried at column i
	
	int i = 0;				
	int j = 0;

	int q; 
	
	// Set to start values
	for (i; i<k; i++){
		S[I+i] = -1;
		D[I+i] = 0;
		for (j;j<k;j++){
			N[I+i][j] = 0;
		}
		j = 0;
	}

	i = 0;

	int iter = 0;
	
	while (iter < 1000){
	
		q = (generate(globalState, i) * k);	// Generate random number
		
		if (D[I+q] == 0 & N[I+i][q] == 0){ 		// Row clear and not tried before
			N[I+i][q] = 1;
			if (checkDiagonals(q,i,S)==1){	// If no attacking queens in diagonal
				S[I+i] = q;			// it can proceed
				D[I+q] = 1;
				i++;
				if (i==k){			// Finished!
					break;
				}
			}
		}
		if (sum(N[I+i],k) + sum(D,k) == k){
			D[I+S[I+i-1]] = 0;
			S[I+i-1] = -1;
			j = 0;
			for (j;j<k;j++){		// Reset N
				N[I+i][j] = 0;
			}		
			i--;				// Backtrack
		}
		iter++;
	}

	//for (int p=0;p<k;p++){
	//	solution[I+p] = S[I+p];
	//}
}

int main() 
{
	size_t avail;
	size_t total;
	cudaMemGetInfo( &avail, &total );
	size_t used = total - avail;
	cout << "Device memory used: " << used << endl;

	curandState* devStates;
	cudaMalloc ( &devStates, k*sizeof( curandState ) );

	// Initialze seeds
	setup_kernel <<< NUM_BLOCKS, NUM_THREADS>>> ( devStates,unsigned(time(NULL)) );

	int solution_host[k*NUM_BLOCKS*NUM_THREADS];
	int* solution_dev;

	cudaMalloc((void**) &solution_dev, sizeof(int)*k*NUM_BLOCKS*NUM_THREADS);
	 
	kernel<<<NUM_BLOCKS,NUM_THREADS>>> (solution_dev, devStates);
	cudaMemcpy(solution_host, solution_dev, sizeof(int)*k*NUM_BLOCKS*NUM_THREADS, cudaMemcpyDeviceToHost);
	
	for (int l=0;l<NUM_BLOCKS*NUM_THREADS;l++){
		if (solution_host[l*k+k-1]!=-1){
			for (int p=0;p<k;p++){
				printf("%d ", solution_host[l*k+p]);
			}
			printf("\n");
		}

	}

	// Free memory
	cudaFree(devStates);
	cudaFree(solution_dev);
	
	cudaDeviceReset(); // Tried to fix memory leakage
	
	return 0;
}
