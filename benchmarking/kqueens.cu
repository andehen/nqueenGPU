#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define NUM_BLOCKS $NUM_BLOCKS
#define NUM_THREADS $NUM_THREADS
#define K $K
#define MAX_ITER $MAX_ITER 

using namespace std;

__device__ int checkDiagonals(int q,int i, int* S)
{
	int I = blockIdx.x*NUM_THREADS*K + threadIdx.x*K;
	int j = 1;
	for (j; j<=i; j++){
		if (S[I+i-j] == q-j | S[I+i-j] == q+j){
			return 0;
		} 		
	}	
	return 1;
}

__device__ int sum(int row[], int len)
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
	int id = blockIdx.x*NUM_BLOCKS + threadIdx.x;
	curand_init ( seed, id, 0, &state[id] );
}

__global__ void kernel(int* S, curandState* globalState)
{
	int I = blockIdx.x*NUM_THREADS*K + threadIdx.x*K;
	int ind = blockIdx.x*NUM_BLOCKS + threadIdx.x;
	int D[K];				
	int N[K][K];			
	
	int i = 0;				
	int j = 0;

	int q; 
	
	for (i; i<K; i++){
		S[I+i] = -1;
		D[i] = 0;
		for (j;j<K;j++){
			N[i][j] = 0;
		}
		j = 0;
	}

	i = 0;

	int iter = 0;

	curandState localState = globalState[ind];
	
	while (iter < MAX_ITER){
	
		q = curand_uniform( &localState ) * K;
		
		if (D[q] == 0 & N[i][q] == 0){ 		 
			N[i][q] = 1;
			if (checkDiagonals(q,i,S)==1){	
				S[I+i] = q;			
				D[q] = 1;
				i++;
				if (i==K){			
					break;
				}
			}
		}
		if (sum(N[i],K) + sum(D,K) == K){
			D[S[I+i-1]] = 0;
			S[I+i-1] = -1;
			j = 0;
			for (j;j<K;j++){	
				N[i][j] = 0;
			}		
			i--;				
		}
		iter++;
	}
}

int main() 
{
	curandState* devStates;
	cudaMalloc ( &devStates, K*sizeof( curandState ) );

	setup_kernel <<< NUM_BLOCKS, NUM_THREADS>>> ( devStates,unsigned(time(NULL)) );

	int solution_host[K*NUM_BLOCKS*NUM_THREADS];
	int* solution_dev;

	cudaMalloc((void**) &solution_dev, sizeof(int)*K*NUM_BLOCKS*NUM_THREADS);

	clock_t begin = clock();
	kernel<<<NUM_BLOCKS,NUM_THREADS>>> (solution_dev, devStates);
	cudaMemcpy(solution_host, solution_dev, sizeof(int)*K*NUM_BLOCKS*NUM_THREADS, cudaMemcpyDeviceToHost);
	clock_t end = clock();
	
	double elapsed_sec = double(end - begin)/(CLOCKS_PER_SEC/1000);

	cout << elapsed_sec << endl;

	int solution_count = 0;
	for (int l=0;l<NUM_BLOCKS*NUM_THREADS;l++){
		if (solution_host[l*K+K-1]!=-1){
			solution_count++;
		}
	}
	cout << solution_count << endl;

	cudaFree(devStates);
	cudaFree(solution_dev);
	
	cudaDeviceReset(); 

	return 0;


}
