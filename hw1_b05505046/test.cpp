#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include<iostream>
#include "hmm.h"
using namespace std;

int main(int argc, char *argv[]) {
	if (argc != 4) {
		fprintf(stderr, "¥Îªk: %s  ./test  modellist.txt  testing_data.txt  result.txt\n", argv[0]);
		exit(1);
	}
	char model_list[65];
	char test[65];
	char result[65];
	strcpy(model_list, argv[1]);   
	strcpy(test, argv[2]);     
	strcpy(result, argv[3]);    
    int timlong = 50;     // length of each sequence 
    int num = 2500;  // number of testing sequences
	int state_num = 6;    // state number
	int all[num][timlong];
	HMM hmms[5];
	load_models(model_list, hmms, 5);
	FILE *fil1 = fopen(test, "rb");
	for (int i = 0; i < num; i++) {
		char tran[timlong];
		fscanf(fil1, "%s", tran);
		for (int j = 0; j < timlong; j++)
			all[i][j] = tran[j] - 65;
	}
	fclose(fil1);

	fil1 = fopen(result, "wb");
	for (int n = 0; n < num; n++) {//run all the sample to find the best model
		double maxp = 0.;
		int bestmodel = 0;
		for (int m = 0; m < 5; m++) {
			//start the viterbi algorithm
			double delta[state_num][timlong];
			for (int i = 0; i < state_num; i++)
			{
				for (int j = 0; j < timlong; j++)
				{
					delta[i][j] = 0;
				}
			}
			// initialize delta
			for (int i = 0; i < state_num; i++)
				delta[i][0] = hmms[m].initial[i] * hmms[m].observation[all[n][0]][i];
			//continue to find the all delta
			for (int t = 1; t < timlong; t++) {
				for (int j = 0; j < state_num; j++) {
					for (int i = 0; i < state_num; i++) {
						double tmp = delta[i][t - 1] * hmms[m].transition[i][j];
						if (tmp > delta[j][t])
							delta[j][t] = tmp;
					}
					delta[j][t] *= hmms[m].observation[all[n][t]][j];
				}
			}
			double max_P = 0.;
			for (int i = 0; i < state_num; i++)
				if (delta[i][timlong - 1] > max_P)
					max_P = delta[i][timlong - 1];

			if (max_P > maxp) {
				maxp = max_P;
				bestmodel = m+1;
			}
		}
		// print the result to the specified file
		fprintf(fil1, "model_0%d.txt %e\n", bestmodel, maxp);
	}
	fclose(fil1);
}