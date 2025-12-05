#pragma once
#include<iostream>
#include<cuda.h>
#include<cmath>
#include<random>
#include<cassert>
using namespace std;

void matMulTCpu(float** h_A, float** h_B, float** h_C, int M, int N, int K)
{
    float pSum = 0.0;
    for(int i = 0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            pSum = 0.0;
            for(int inLoop = 0; inLoop<K; inLoop++)
            {
                pSum+= (h_A[i][inLoop]*h_B[j][inLoop]);
            }
            h_C[i][j] = pSum;
        }
    }
}

void matMulCpu(float** h_A, float** h_B, float** h_C, int M, int N, int K)
{
    float pSum = 0.0;
    for(int i = 0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            pSum = 0.0;
            for(int inLoop = 0; inLoop<K; inLoop++)
            {
                pSum+= (h_A[i][inLoop]*h_B[inLoop][j]);
            }
            h_C[i][j] = pSum;
        }
    }
}

void softmaxCpu(float** h_A, float** h_C, int M, int N)
{
    float sum = 0.0;
    float globalMax = -INFINITY;
    float val;

    for(int i = 0; i<M; i++)
    {
        sum = 0.0;
        globalMax = -INFINITY;

        //Pass 1
        for(int j=0; j<N; j++)
        {
            val = h_A[i][j];
            if(val > globalMax)
            {
                globalMax = val;
            }
            //printf("%f \t",globalMax);
        }

        //Pass 2
        for(int j=0; j<N; j++)
        {
            val = h_A[i][j];
            sum += expf(val - globalMax);
        }

        //Pass 3
        for(int j=0; j<N; j++)
        {
            val = h_A[i][j];
            h_C[i][j] = expf(val-globalMax)/sum;
            //printf("%f \t",h_C[i][j]);
        }
        if(i == 7)
        printf("\n");
    }
}

float** assignHostSpace(int rows, int cols)
{
    float** hostArr;
    hostArr = new float*[rows];
    for(int i=0; i<rows; i++)
    {
        hostArr[i] = new float[cols];
        for(int j=0; j<cols; j++)
        {
            hostArr[i][j] = 0;
        }
    }
    return hostArr;
}

void assignHostValues(float** hostArr, int rows, int cols)
{
    mt19937 gen(2026);  // fixed seed for determinism
    uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
    for(int i=0;i<rows;i++)
    {
        for(int j=0; j<cols; j++)
        {
            hostArr[i][j] = dist(gen);
        }
    }
}

void mismatch2D(float** cpuArr, float** gpuArr, int row, int col)
{
    int flag = 1;
    const float epsilon = 1e-4f; // 5 decimal places tolerance
    for(int i=0; i<row; i++)
    {
        for(int j = 0; j<col; j++)
        {
            if(fabsf(cpuArr[i][j] - gpuArr[i][j]) > epsilon)
            {
                flag = 0;
                printf("Mismatch at : (%d,%d), CPU: %f ; GPU: %f \n",i,j,cpuArr[i][j],gpuArr[i][j]);
            }
        }
    }
    if(flag == 1)
        printf("Success \n");
}

void attentionCpu(float** Q, float** K, float** V, float** O, float N, float d)
{
    float** S;
    S = assignHostSpace(N, N);
    matMulTCpu(Q, K, S, N, N, d);
    /*printf("CPU : \n");
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
        {
            printf("%f \t", S[i][j]);
        }
        printf("\n");
    }*/
    float** P;
    P = assignHostSpace(N, N);
    softmaxCpu(S, P, N, N);
    matMulCpu(P, V, O, N, d, N);
}