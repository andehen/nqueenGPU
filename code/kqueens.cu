#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define k 10 // set problem size 

using namespace std;

__device__ float generate(curandState* globalState, int ind2) 
// Function to generate random number in thread
{
	int ind = threadIdx.x;
	curandState localState = globalState[ind];
	float rnd = curand_uniform( &localState );
	globalState[ind] = localState;
	return rnd;
}

__device__ int checkDiagonals(int q,int i, int S[])
// Returns 1 if no queen in diagonal, else 0
{
	int j = 1;
	for (j; j<=i; j++){
		if (S[i-j] == q-j | S[i-j] == q+j){
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
{
	int id = threadIdx.x;
	curand_init ( seed, id, 0, &state[id] );
}

__global__ void kernel(int* solution, curandState* globalState)
{

	// Initialize varaibles
	int S[k]; 				// Holds current solution
	int D[k];				// Rows where queens is placed
	int N[k][k];				// Positions tried at column i
	
	int i = 0;				
	int j = 0;

	int q; 
	
	// Set to start values
	for (i; i<k; i++){
		S[i] = -1;
		D[i] = 0;
		for (j;j<k;j++){
			N[i][j] = 0;
		}
		j = 0;
	}

	i = 0;
	
	while (S[k-1] == -1){
	
		q = (generate(globalState, i) * k);	// Generate random number
		
		if (D[q] == 0 & N[i][q] == 0){ 		// Row clear and not tried before 
			N[i][q] = 1;
			if (checkDiagonals(q,i,S)==1){	// If no attacking queens in diagonal
				S[i] = q;			// it can proceed
				D[q] = 1;
				i++;
				if (i==k){			// Finished!
					break;
				}
			}
		}
		if (sum(N[i],k) + sum(D,k) == k){
			D[S[i-1]] = 0;
			S[i-1] = -1;
			j = 0;
			for (j;j<k;j++){		// Reset N
				N[i][j] = 0;
			}		
			i--;				// Backtrack
		}
	}
	// For now, just print solution for each thread for debugging
	printf("Sol from block %d, thread %d: ", blockIdx, threadIdx);
	for (int l=0;l<k;l++){
		printf("%d ", S[l]);
	}
	printf("\n");
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
	setup_kernel <<< 10, 2 >>> ( devStates,unsigned(time(NULL)) );

	int solution2[k];
	int* solution3;

	cudaMalloc((void**) &solution3, sizeof(int)*k);
	 
	kernel<<<10,2>>> (solution3, devStates);
	cudaMemcpy(solution2, solution3, sizeof(int)*k, cudaMemcpyDeviceToHost);
	
	// Free memory
	cudaFree(devStates);
	cudaFree(solution3);
	
	cudaDeviceReset(); // Tried to fix memory leakage
	
	return 0;
}