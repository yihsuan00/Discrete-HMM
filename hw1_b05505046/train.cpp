#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include "hmm.h"
using namespace std;
int getfilesize(char* filename)
{
	ifstream is;
	is.open(filename, ios::binary);
	is.seekg(0, ifstream::end);
	int size = is.tellg();
	is.seekg(0);
	is.close();
	return size;
}
void trainhmm(HMM *hmm, char *train_set)//Ūmodel
{
    int seqlen = 50;     // length of sequence in seq_modellist
	double data_num = 10000; // number of sequences (samples)
    int state_num = 6;    // hmm->state_num
    int observ_num = 6;   // hmm->observ_num
	double p[state_num] = { 0. };
	//double transienta[state_num][state_num] = { 0. };
	double acumulate_gama[seqlen][state_num];
	//double transientb[state_num][state_num] = { 0. };
	double accumulate_e[state_num][state_num];
	double obgama[state_num][state_num];
	for (int i = 0; i < seqlen; i++)
	{
		for (int j = 0; j < state_num; j++)
		{
			acumulate_gama[i][j] = 0;
		}
	}
	for (int i = 0; i < state_num; i++)
	{
		for (int j = 0; j < state_num; j++)
		{
			accumulate_e[i][j] = 0;
			obgama[i][j] = 0;
		}
	}
	ifstream openfile;
	if (!openfile)
	{
		printf("wrong in opening file\n");
		return;
	}
	openfile.open(train_set, ios::binary);
	int size = getfilesize(train_set);
	char* fileContent = new char[size];
	openfile.read(fileContent, size);
	int times = 0;
	int col = 0;
	char all[10000][50];
	for (int i = 0; i < size; i++)
	{
		if (int(fileContent[i] - 'A') < 0)
		{
			times++;
			col = 0;
			continue;
		}
		all[times][col] = int(fileContent[i] - 'A');
		col++;
	}
	openfile.close();
	for(int r=0;r<data_num;r++)
	{
		double alpha[seqlen][state_num];
		double beta[seqlen][state_num];
		double gama[seqlen][state_num];
		double e[seqlen][state_num][state_num];
		for (int i = 0; i < seqlen; i++)
		{
			for (int j = 0; j < state_num; j++)
			{
				alpha[i][j]=0;
				beta[i][j]=0;
				gama[i][j]=0;
			}
		}
		for (int x = 0; x < seqlen; x++)
		{
			for (int i = 0; i <state_num; i++)
			{
				for (int j = 0; j < state_num; j++)
				{
					e[x][i][j] = 0;
				}
			}
		}
		
		//initialize alpha
		for (int i = 0; i < 6; i++)
		{
			alpha[0][i] = hmm->initial[i] * hmm->observation[all[r][0]][i];
			//printf("alpha[0][%d]:%e\n", i, alpha[0][i]);
		}
		//forward 
		for (int i = 1; i < 50; i++)
		{
			for (int j = 0; j < 6; j++)
			{
				for (int k = 0; k < 6; k++)
				{
					alpha[i][j] += (alpha[i - 1][k] * hmm->transition[k][j]);
				}
				alpha[i][j] *= hmm->observation[all[r][i]][j];
			}
		}
		//initialize beta
		for (int i = 0; i < 6; i++)
		{
			beta[49][i] = 1;
		}
		//backward 
		for (int i = 1; i < 50; i++)
		{
			for (int j = 0; j < 6; j++)
			{
				for (int k = 0; k < 6; k++)
				{
					beta[49-i][j] += (beta[50-i][k] * hmm->transition[j][k]*hmm->observation[all[r][50-i]][k]);	
				}
			}
		}
		//gamma
		for (int i = 0; i < 50; i++)
		{
			double sum = 0;
			for (int j = 0; j < 6; j++)
			{
				sum +=(alpha[i][j] * beta[i][j]);
				
			}
			for (int j = 0; j < 6; j++)
			{
				gama[i][j] = alpha[i][j] * beta[i][j] / sum;
			}
		}
		
		//epsilon
		for (int i = 0; i < 49; i++)
		{
			double sume=0;
			for (int j = 0; j < 6; j++)
			{
				for (int k = 0; k < 6; k++)
				{
					sume += alpha[i][j] * beta[i+1][k] * hmm->observation[all[r][i+1]][k] * hmm->transition[j][k];
				}
			}
			for (int j = 0; j < 6; j++)
			{
				for (int k = 0; k < 6; k++)
				{
					e[i][j][k] = alpha[i][j] * beta[i + 1][k] * hmm->observation[all[r][i + 1]][k] * hmm->transition[j][k] / sume;
				}
			}
		}
		//acumulate gama,epsilon
		for (int i = 0; i < 6; i++)
		{
			for (int j = 0; j < 50; j++)
			{
				acumulate_gama[j][i] += gama[j][i];
			}
			for (int k = 0; k < 6; k++)
			{
				for (int j = 0; j < 49; j++)
				{
					accumulate_e[i][k]+=e[j][i][k];
				}
			}
		}
		//observation gama
		for (int i = 0; i < 49; i++)
		{
			for (int j = 0; j < 6; j++)
			{
				obgama[all[r][i]][j] += gama[i][j];
			}
		}
		//update variable
		for (int i = 0; i < 6; i++)
		{
			p[i] = p[i] + gama[0][i];
		}
			/*for (int j = 0; j < 6; j++)
			{
				transienta[i][j] += accumulate_e[i][j] / acumulate_gama[i];
				//printf("%e ", transienta[i][j]);
				acumulate_gama[i] += gama[49][i];
				transientb[i][j] += obgama[i][j] / acumulate_gama[i];
			}
			//printf("\n");
		}*/
}
	//change parameters
	double gamma_sum[state_num];
	for (int t = 0; t < seqlen - 1; t++)
		for (int i = 0; i < state_num; i++)
			gamma_sum[i] += acumulate_gama[t][i];
	for (int i = 0; i < state_num; i++)
		for (int j = 0; j < state_num; j++)
			hmm->transition[i][j] = accumulate_e[i][j] / gamma_sum[i];
	for (int i = 0; i < 6; i++)
	{
		hmm->initial[i] = p[i] / data_num;
	}
	for (int i = 0; i < state_num; i++)
		gamma_sum[i] += acumulate_gama[49][i];
	for (int j = 0; j < state_num; j++)
		for (int k = 0; k < observ_num; k++)
			hmm->observation[k][j] = obgama[k][j] / gamma_sum[j];
}
int main(int argc, char* argv[])
{
	if (argc != 5) {
		fprintf(stderr, "�Ϊk: %s ./train  iteration  model_init.txt  seq_model_01.txt model_01.txt \n", argv[0]);
		exit(1);
	}
	int iter= atoi(argv[1]);
	char  init_file[65];    
	char  train_file[65];   
	char  dump_file[65];    
	sscanf(argv[2],"%s",init_file);
	sscanf(argv[3],"%s", train_file);
	sscanf(argv[4], "%s",dump_file );
	HMM hmm;
	// initialize HMM for further update
	loadHMM(&hmm, init_file);
	printf("hmmob	#%e   \n", hmm.observation[1][2]);
	// train HMM with trainging data (seq_model_01~05.txt)
	
	for (int i = 0; i < iter; i++) {
      trainhmm(&hmm, train_file);
	  printf("iter: %d\n", i);
	}
	// dump the training result (model parameters)
	FILE *fp = fopen(dump_file, "wb");
	dumpHMM(fp, &hmm);
	fclose(fp);
}