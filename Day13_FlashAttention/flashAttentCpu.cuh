
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
        //printf("\n");
    }
}

void attentionCpu(float** Q, float** K, float** V, float** O, float N, float d)
{
    float** S;
    allocate2d(S, N, N);
    matMulTCpu(Q, K, S, int N, int N, int d);
    float** P;
    allocate2d(P, N, N);
    softmaxCpu(S, P, N, N);
    matMulTCpu(P, V, O, int N, int d, int N);
}