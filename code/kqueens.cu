#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define k 10

using namespace std;

__device__ float generate(curandState* globalState, int ind2 ) 
{
    int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float rnd = curand_uniform( &localState );
    globalState[ind] = localState;
    return rnd;
}
__device__ int diagonalsOK(int q,int i, int S[]){
	int j = 1;
	for (j; j<=i; j++){
		if (S[i-j] == q-j | S[i-j] == q+j){
			return 0;
		} 		
	}	
	return 1;
}

__device__ int sum(int row[], int len){
	int i = 0;
	int s = 0;
	for (i; i<len; i++){
		s += row[i];
	}
	return s;
}

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

__global__ void kernel(int* solution, curandState* globalState)
{

	// Initialize varaibles
	int S[k];
	int D[k];
	int N[k][k];
	
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
	
        q = (generate(globalState, i) * k);
		
		if (D[q] == 0 & N[i][q] == 0){ // Row clear and not tried before 
			N[i][q] = 1;
			if (diagonalsOK(q,i,S)==1){
				S[i] = q;
				D[q] = 1;
				i++;
				if (i==k){ // Finished!
					break;
				}
			}
		}
		if (sum(N[i],k) + sum(D,k) == k){
			D[S[i-1]] = 0;
			S[i-1] = -1;
			// Set N[i] = 0;
			j = 0;
			for (j;j<k;j++){
				N[i][j] = 0;
			}		
			i--; // Backtrack
		}
	}
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

    // setup seeds
    setup_kernel <<< 1, k >>> ( devStates,unsigned(time(NULL)) );
   
	int solution2[k];
	int* solution3;

	cudaMalloc((void**) &solution3, sizeof(int)*k);
	 
	kernel<<<2,1>>> (solution3, devStates);
	cudaMemcpy(solution2, solution3, sizeof(int)*k, cudaMemcpyDeviceToHost);
	
	cudaFree(devStates);
	cudaFree(solution3);
    cudaDeviceReset();
    return 0;
}