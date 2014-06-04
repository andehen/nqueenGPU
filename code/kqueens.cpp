#include <time.h>
#include <stdlib.h>

void printS( int S[], int k){
	int i = 0;
	for (i; i<k-1; i++){
		printf("%d ", S[i]); 
	}
	printf("%d \n", S[k-1]);
}	


int diagonalsOK(int q,int i, int S[]){
	int j = 1;
	for (j; j<=i; j++){
		if (S[i-j] == q-j | S[i-j] == q+j){
			return 0;
		} 		
	}	
	return 1;
}

int sum(int row[], int len){
	int i = 0;
	int s = 0;
	for (i; i<len; i++){
		s += row[i];
	}
	return s;
}

int main (){
	// Initialize varaibles
	int k = 30;
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

	// Prepare values and rnd num
	srand(time(NULL));
	i = 0;
	
//	S[0] = 1;
//	S[1] = 3;
//	q = 0;
//	i = 2;
//	int t = diagonalsOK(q,i,S);
//	printf("%d",t);

	//int T[4] = {1,0,1,0};
	//printf("T: %d\n",sum(T,4)); 

	// Solve puzzle
	while (S[k-1] == -1){
	//	printf("i: %d\n", i);
	//	printf("S: ");
	//	printS(S,k);
	//	printf("D: ");
	//	printS(D,k);
	//	printf("N: ");
	//	printS(N[i],k);
	//	printf("q: ");
	
		q = rand() % k;
	
	//	printf("%d\n",q);
		
		if (D[q] == 0 & N[i][q] == 0){ // Row clear and not tried before 
			N[i][q] = 1;
			if (diagonalsOK(q,i,S)==1){
				//printf("Diagonals ok! Next iteration\n");
				S[i] = q;
				D[q] = 1;
				i++;
				if (i==k){ // Finished!
					break;
				}
			}
		}
		if (sum(N[i],k) + sum(D,k) == k){
			//printf("Backtrack\n");
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
	printf("Terminated!\n");
	printS(S,k);
}
