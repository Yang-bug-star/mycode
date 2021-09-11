#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <sys/time.h>
#include <omp.h>

using namespace std;

const int minmum_size = 2, maximal_size = 2048;
int thread_count;

#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}

// 生成0~10的随机数
float random_element(){
    return rand() / double(RAND_MAX) * 10;
}

void print_matrix(const float* mat, int M, int N){
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            cout << mat[i * N + j] << " ";
        }
        cout << endl;
    }
}

void Create_RandomMatrix(float* mat, int M, int N){
    for (int i = 0; i < M * N; ++i)
        mat[i] = random_element();
}

inline int prePos(int M, int N, int index)
{   //找到前驱
    return (index % M) * N + index / M; 
}
 
inline int nextPos(int M, int N, int index)
{  //找到后继
    return (index % N) * M + index / N;
}

void transpose(float* mat, int M, int N)
{
    int len = M * N;
    int pre, next;

    for(int i = 0; i< len; i++)
    {
        pre = prePos(M, N, i);
        next = nextPos(M, N, i); 
        //指针向两方向同时移动
        while(pre > i && next > i && pre != next && prePos(M, N, pre) != next) 
        {
            pre = prePos(M, N, pre);
            next = nextPos(M, N, next); 
        }
        if(pre < i || next < i)//此环已被处理过
            continue;

        int cur =i;
        float val = mat[i];

        pre = prePos(M, N, cur);
        while(pre != i)//移动环中的元素
        {
            mat[cur] = mat[pre];
            cur = pre;
            pre = prePos(M, N, cur);
        }
        mat[cur] = val;
    } 
}

void GEMM_Serial_1(const float* A, const float* B, float* C, int M, int N, int K, const int strideA, const int strideB){
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < K; ++j){
            C[i * K + j] = 0.0;
            for (int n = 0; n < N; ++n){
                C[i * K + j] += A[i * strideA + n] * B[n * strideB + j];
            }
        }
    }
}
 
void GEMM_Serial_2(const float* A, const float* B, float* C, int M, int N, int K, const int strideA, const int strideB){
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < K; ++j){
            C[i * K + j] = 0.0;
            for (int n = 0; n < N; ++n){
                C[i * K + j] += A[i * strideA + n] * B[j * strideB + n];
            }
        }
    }
}

void GEMM_Parallel_1(const float* A, const float* B, float* C, int M, int N, int K, const int strideA, const int strideB){
    # pragma omp for  nowait
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < K; ++j){
            C[i * K + j] = 0.0;
            for (int n = 0; n < N; ++n){
                C[i * K + j] += A[i * strideA + n] * B[n * strideB + j];
            }
        }
    }
}

void GEMM_Parallel_2(const float* A, const float* B, float* C, int M, int N, int K, const int strideA, const int strideB){
    # pragma omp for  nowait
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < K; ++j){
            C[i * K + j] = 0.0;
            for (int n = 0; n < N; ++n){
                C[i * K + j] += A[i * strideA + n] * B[j * strideB + n];
            }
        }
    }
}

void Strassen_use_GEMM_Serial_1(const float* A, const float* B, float* C, const int M, const int N, const int K, const int strideA, const int strideB)
{
	if (M * N < 64 * 64 || N * K < 64 * 64 || M % 2 || N % 2 || K % 2)   return GEMM_Serial_1(A, B, C, M, N, K, strideA, strideB);
	memset(C, 0, M * K * sizeof(float));

    int new_M, new_N, new_K;
    new_M = M / 2;
    new_N = N / 2;
    new_K = K / 2;

    //P1 = A11*S1 = A11*(B12-B22)
    float* P1 = new float[new_M * new_K];
	{
		//P1_1 = (B12-B22)
        float* P1_1 = new float[new_N * new_K];

		for (int i = 0; i < new_N; i++)
		{
			for (int j = new_K; j < K; j++)
				P1_1[i * new_K + j - new_K] = B[i * strideB + j] - B[i * strideB + j + new_N * strideB];
		}

		Strassen_use_GEMM_Serial_1(A, P1_1, P1, new_M, new_N, new_K, 
            strideA, new_K);

        delete[] P1_1;
	}

    //P2 = S2*B22 = (A11+A12)*B22
    float* P2 = new float[new_M * new_K];
	{
		//P2_0 = (A11+A12)
        float* P2_0 = new float[new_M * new_N];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P2_0[i * new_N + j] = A[i * strideA + j] + A[i * strideA + j + new_N];
		}

		Strassen_use_GEMM_Serial_1(P2_0, B + new_N * strideB + new_K, P2, new_M, new_N, new_K,
			new_N, strideB);

         delete[] P2_0;
	}

    //P3 = S3*B11 = (A21+A22)*B11
    float* P3 = new float[new_M * new_K];
	{
		//P3_0 = (A21+A22) 
        float* P3_0 = new float[new_M * new_N];

		for (int i = new_M; i < M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P3_0[(i - new_M) * new_N + j] = A[i * strideA + j] + A[i * strideA + j + new_N];
		}
		//M2_2 = B11
		Strassen_use_GEMM_Serial_1(P3_0, B, P3, new_M, new_N, new_K,
			new_N, strideB);
        delete[] P3_0;
	}

    //P4 = A22*S4 = A22*(B21-B11)
    float* P4 = new float[new_M * new_K];
	{
		//P4_1 = (B21-B11)
        float* P4_1 = new float[new_N * new_K];
	
		for (int i = 0; i < new_N; i++)
		{
			for (int j = 0; j < new_K; j++)
				P4_1[i * new_K + j] = B[i * strideB + j + new_N * strideB] - B[i * strideB + j];
		}
		Strassen_use_GEMM_Serial_1(A + new_M * strideA + new_N, P4_1, P4, new_M, new_N, new_K,
			strideA, new_K);

        delete[] P4_1;
	}

	//P5 = S5*S6 = (A11+A22)*(B11+B22)
    float* P5 = new float[new_M * new_K];
	{
		//P5_0= (A11+A22) 
        //P5_1 = (B11+B22)
        float* P5_0 = new float[new_M * new_N];
        float* P5_1 = new float[new_N * new_K];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P5_0[i * new_N + j] = A[i * strideA + j] + A[i * strideA + j + new_M * strideA + new_N];
		}
		
		for (int i = 0; i < new_N; i++)
		{
			for (int j = 0; j < new_K; j++)
				P5_1[i * new_K + j] = B[i * strideB + j] + B[i * strideB + j + new_N * strideB + new_K];
		}

		Strassen_use_GEMM_Serial_1(P5_0, P5_1, P5, new_M, new_N, new_K,
            new_N, new_K);

        delete[] P5_0;
        delete[] P5_1;
	}

    //P6 = S7*S8 = (A12-A22)*(B21+B22)
    float* P6 = new float[new_M * new_K];
	{
		//P6_0 = (A12-A22) 
        //P6_1 = (B21+B22)
        float* P6_0 = new float[new_M * new_N];
		float* P6_1 = new float[new_N * new_K];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = new_N; j < N; j++)
				P6_0[i * new_N + j - new_N] = A[i * strideA + j] - A[i * strideA + j + new_M * strideA];
		}

		for (int i = new_N; i < N; i++)
		{
			for (int j = 0; j < new_K; j++)
				P6_1[(i - new_N) * new_K + j] = B[i * strideB + j] + B[i * strideB + j + new_K];
		}

		Strassen_use_GEMM_Serial_1(P6_0, P6_1, P6, new_M, new_N, new_K,
			new_N, new_K);

        delete[] P6_0;
        delete[] P6_1;
	}	

	//P7 = S9*S10 = (A11-A21)*(B11+B12) 
    float* P7 = new float[new_M * new_K];
	{
		//P7_0 = (A11-A21) 
        //P7_1 = (B11+B12) 
        float* P7_0 = new float[new_M * new_N];
        float* P7_1 = new float[new_N * new_K];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P7_0[i * new_N + j] = A[i * strideA + j] - A[i * strideA + j + new_M * strideA];
		}

		for (int i = 0; i < new_N; i++)
		{
			for (int j = 0; j < new_K; j++)
				P7_1[i * new_K + j] = B[i * strideB + j] + B[i * strideB + j + new_K];
		}

		Strassen_use_GEMM_Serial_1(P7_0, P7_1, P7, new_M, new_N, new_K,
			new_N, new_K);

        delete[] P7_0;
        delete[] P7_1;
	}


	for (int i = 0; i < new_M; i++)
	{
		for (int j = 0; j < new_K; j++)
		{
			const int idx = i * new_K + j;
			//C11 = M1+M4-M5+M7
			C[i * K + j] = P5[idx] + P4[idx] - P2[idx] + P6[idx];
			//C12 = M3+M5
			C[i * K + j + new_K] = P1[idx] + P2[idx];
			//C21 = M2+M4
			C[(i + new_M) * K + j] = P3[idx] + P4[idx];
			//C22 = M1-M2+M3+M6
			C[(i + new_M) * K + j + new_K] = P5[idx] + P1[idx] - P3[idx] - P7[idx];
		}
	}
    delete[] P1;
    delete[] P2;
    delete[] P3;
    delete[] P4;
    delete[] P5;
    delete[] P6;
    delete[] P7;
}

void Strassen_use_GEMM_Serial_2(const float* A, const float* B, float* C, const int M, const int N, const int K, const int strideA, const int strideB)
{
	if (M * N < 64 * 64 || N * K < 64 * 64 || M % 2 || N % 2 || K % 2)   return GEMM_Serial_2(A, B, C, M, N, K, strideA, strideB);
	memset(C, 0, M * K * sizeof(float));

    int new_M, new_N, new_K;
    new_M = M / 2;
    new_N = N / 2;
    new_K = K / 2;

    //P1 = A11**(B21-B22), **new operator   
    float* P1 = new float[new_M * new_K];
	{
		//P1_1 = (B21-B22)
        float* P1_1 = new float[new_K * new_N];

		for (int i = new_K; i < K; i++)
		{
			for (int j = 0; j < new_N; j++)
				P1_1[(i - new_K) * new_N + j] = B[i * strideB + j] - B[i * strideB + j + new_N];
		}

		Strassen_use_GEMM_Serial_2(A, P1_1, P1, new_M, new_N, new_K, 
            strideA, new_N);

        delete[] P1_1;
	}

    //P2 = (A11+A12)**B22
    float* P2 = new float[new_M * new_K];
	{
		//P2_0 = (A11+A12)
        float* P2_0 = new float[new_M * new_N];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P2_0[i * new_N + j] = A[i * strideA + j] + A[i * strideA + j + new_N];
		}

		Strassen_use_GEMM_Serial_2(P2_0, B + new_K * strideB + new_N, P2, new_M, new_N, new_K,
			new_N, strideB);

         delete[] P2_0;
	}

    //P3 = (A21+A22)**B11
    float* P3 = new float[new_M * new_K];
	{
		//P3_0 = (A21+A22) 
        float* P3_0 = new float[new_M * new_N];

		for (int i = new_M; i < M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P3_0[(i - new_M) * new_N + j] = A[i * strideA + j] + A[i * strideA + j + new_N];
		}
		//M2_2 = B11
		Strassen_use_GEMM_Serial_2(P3_0, B, P3, new_M, new_N, new_K,
			new_N, strideB);

        delete[] P3_0;
	}

    //P4 = A22**(B12-B11)
    float* P4 = new float[new_M * new_K];
	{
		//P4_1 = (B12-B11)
        float* P4_1 = new float[new_K * new_N];
	
		for (int i = 0; i < new_K; i++)
		{
			for (int j = 0; j < new_N; j++)
				P4_1[i * new_N + j] = B[i * strideB + j + new_N] - B[i * strideB + j];
		}

		Strassen_use_GEMM_Serial_2(A + new_M * strideA + new_N, P4_1, P4, new_M, new_N, new_K,
			strideA, new_N);

        delete[] P4_1;
	}

	//P5 = (A11+A22)**(B11+B22)
    float* P5 = new float[new_M * new_K];
	{
		//P5_0= (A11+A22) 
        //P5_1 = (B11+B22)
        float* P5_0 = new float[new_M * new_N];
        float* P5_1 = new float[new_K * new_N];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P5_0[i * new_N + j] = A[i * strideA + j] + A[i * strideA + j + new_M * strideA + new_N];
		}
		
		for (int i = 0; i < new_K; i++)
		{
			for (int j = 0; j < new_N; j++)
				P5_1[i * new_N + j] = B[i * strideB + j] + B[i * strideB + j + new_K * strideB + new_N];
		}

		Strassen_use_GEMM_Serial_2(P5_0, P5_1, P5, new_M, new_N, new_K,
            new_N, new_N);

        delete[] P5_0;
        delete[] P5_1;
	}

    //P6 = (A12-A22)**(B12+B22)
    float* P6 = new float[new_M * new_K];
	{
		//P6_0 = (A12-A22) 
        //P6_1 = (B12+B22)
        float* P6_0 = new float[new_M * new_N];
		float* P6_1 = new float[new_K * new_N];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = new_N; j < N; j++)
				P6_0[i * new_N + j - new_N] = A[i * strideA + j] - A[i * strideA + j + new_M * strideA];
		}

		for (int i = 0; i < new_K; i++)
		{
			for (int j = new_N; j < N; j++)
				P6_1[i * new_N + j - new_N] = B[i * strideB + j] + B[i * strideB + j + new_K * strideB];
		}

		Strassen_use_GEMM_Serial_2(P6_0, P6_1, P6, new_M, new_N, new_K,
			new_N, new_N);

        delete[] P6_0;
        delete[] P6_1;
	}	

	//P7 = (A11-A21)**(B11+B21) 
    float* P7 = new float[new_M * new_K];
	{
		//P7_0 = (A11-A21) 
        //P7_1 = (B11+B21) 
        float* P7_0 = new float[new_M * new_N];
        float* P7_1 = new float[new_K * new_N];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P7_0[i * new_N + j] = A[i * strideA + j] - A[i * strideA + j + new_M * strideA];
		}

		for (int i = 0; i < new_K; i++)
		{
			for (int j = 0; j < new_N; j++)
				P7_1[i * new_N + j] = B[i * strideB + j + new_K * strideB] + B[i * strideB + j];
		}

		Strassen_use_GEMM_Serial_2(P7_0, P7_1, P7, new_M, new_N, new_K,
			new_N, new_N);

        delete[] P7_0;
        delete[] P7_1;
	}


	for (int i = 0; i < new_M; i++)
	{
		for (int j = 0; j < new_K; j++)
		{
			const int idx = i * new_K + j;
			//C11 = P5 + P4 - P2 + P6
			C[i * K + j] = P5[idx] + P4[idx] - P2[idx] + P6[idx];
			//C12 = P1 + P2
			C[i * K + j + new_K] = P1[idx] + P2[idx];
			//C21 = P3 + P4
			C[(i + new_M) * K + j] = P3[idx] + P4[idx];
			//C22 = P5 + P1 - P3 - P7
			C[(i + new_M) * K + j + new_K] = P5[idx] + P1[idx] - P3[idx] - P7[idx];
		}
	}
    delete[] P1;
    delete[] P2;
    delete[] P3;
    delete[] P4;
    delete[] P5;
    delete[] P6;
    delete[] P7;
}

void Strassen_use_GEMM_Parallel_1(const float* A, const float* B, float* C, const int M, const int N, const int K, const int strideA, const int strideB)
{
	if (M * N < 64 * 64 || N * K < 64 * 64 || M % 2 || N % 2 || K % 2)   return GEMM_Parallel_1(A, B, C, M, N, K, strideA, strideB);
	memset(C, 0, M * K * sizeof(float));

    int new_M, new_N, new_K;
    new_M = M / 2;
    new_N = N / 2;
    new_K = K / 2;

    //P1 = A11*S1 = A11*(B12-B22)
    float* P1 = new float[new_M * new_K];
	{
		//P1_1 = (B12-B22)
        float* P1_1 = new float[new_N * new_K];

		for (int i = 0; i < new_N; i++)
		{
			for (int j = new_K; j < K; j++)
				P1_1[i * new_K + j - new_K] = B[i * strideB + j] - B[i * strideB + j + new_N * strideB];
		}

		Strassen_use_GEMM_Parallel_1(A, P1_1, P1, new_M, new_N, new_K, 
            strideA, new_K);

        delete[] P1_1;
	}

    //P2 = S2*B22 = (A11+A12)*B22
    float* P2 = new float[new_M * new_K];
	{
		//P2_0 = (A11+A12)
        float* P2_0 = new float[new_M * new_N];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P2_0[i * new_N + j] = A[i * strideA + j] + A[i * strideA + j + new_N];
		}

		Strassen_use_GEMM_Parallel_1(P2_0, B + new_N * strideB + new_K, P2, new_M, new_N, new_K,
			new_N, strideB);

         delete[] P2_0;
	}

    //P3 = S3*B11 = (A21+A22)*B11
    float* P3 = new float[new_M * new_K];
	{
		//P3_0 = (A21+A22) 
        float* P3_0 = new float[new_M * new_N];

		for (int i = new_M; i < M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P3_0[(i - new_M) * new_N + j] = A[i * strideA + j] + A[i * strideA + j + new_N];
		}
		//M2_2 = B11
		Strassen_use_GEMM_Parallel_1(P3_0, B, P3, new_M, new_N, new_K,
			new_N, strideB);
        delete[] P3_0;
	}

    //P4 = A22*S4 = A22*(B21-B11)
    float* P4 = new float[new_M * new_K];
	{
		//P4_1 = (B21-B11)
        float* P4_1 = new float[new_N * new_K];
	
		for (int i = 0; i < new_N; i++)
		{
			for (int j = 0; j < new_K; j++)
				P4_1[i * new_K + j] = B[i * strideB + j + new_N * strideB] - B[i * strideB + j];
		}
		Strassen_use_GEMM_Parallel_1(A + new_M * strideA + new_N, P4_1, P4, new_M, new_N, new_K,
			strideA, new_K);

        delete[] P4_1;
	}

	//P5 = S5*S6 = (A11+A22)*(B11+B22)
    float* P5 = new float[new_M * new_K];
	{
		//P5_0= (A11+A22) 
        //P5_1 = (B11+B22)
        float* P5_0 = new float[new_M * new_N];
        float* P5_1 = new float[new_N * new_K];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P5_0[i * new_N + j] = A[i * strideA + j] + A[i * strideA + j + new_M * strideA + new_N];
		}
		
		for (int i = 0; i < new_N; i++)
		{
			for (int j = 0; j < new_K; j++)
				P5_1[i * new_K + j] = B[i * strideB + j] + B[i * strideB + j + new_N * strideB + new_K];
		}

		Strassen_use_GEMM_Parallel_1(P5_0, P5_1, P5, new_M, new_N, new_K,
            new_N, new_K);

        delete[] P5_0;
        delete[] P5_1;
	}

    //P6 = S7*S8 = (A12-A22)*(B21+B22)
    float* P6 = new float[new_M * new_K];
	{
		//P6_0 = (A12-A22) 
        //P6_1 = (B21+B22)
        float* P6_0 = new float[new_M * new_N];
		float* P6_1 = new float[new_N * new_K];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = new_N; j < N; j++)
				P6_0[i * new_N + j - new_N] = A[i * strideA + j] - A[i * strideA + j + new_M * strideA];
		}

		for (int i = new_N; i < N; i++)
		{
			for (int j = 0; j < new_K; j++)
				P6_1[(i - new_N) * new_K + j] = B[i * strideB + j] + B[i * strideB + j + new_K];
		}

		Strassen_use_GEMM_Parallel_1(P6_0, P6_1, P6, new_M, new_N, new_K,
			new_N, new_K);

        delete[] P6_0;
        delete[] P6_1;
	}	

	//P7 = S9*S10 = (A11-A21)*(B11+B12) 
    float* P7 = new float[new_M * new_K];
	{
		//P7_0 = (A11-A21) 
        //P7_1 = (B11+B12) 
        float* P7_0 = new float[new_M * new_N];
        float* P7_1 = new float[new_N * new_K];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P7_0[i * new_N + j] = A[i * strideA + j] - A[i * strideA + j + new_M * strideA];
		}

		for (int i = 0; i < new_N; i++)
		{
			for (int j = 0; j < new_K; j++)
				P7_1[i * new_K + j] = B[i * strideB + j] + B[i * strideB + j + new_K];
		}

		Strassen_use_GEMM_Parallel_1(P7_0, P7_1, P7, new_M, new_N, new_K,
			new_N, new_K);

        delete[] P7_0;
        delete[] P7_1;
	}


	for (int i = 0; i < new_M; i++)
	{
		for (int j = 0; j < new_K; j++)
		{
			const int idx = i * new_K + j;
			//C11 = M1+M4-M5+M7
			C[i * K + j] = P5[idx] + P4[idx] - P2[idx] + P6[idx];
			//C12 = M3+M5
			C[i * K + j + new_K] = P1[idx] + P2[idx];
			//C21 = M2+M4
			C[(i + new_M) * K + j] = P3[idx] + P4[idx];
			//C22 = M1-M2+M3+M6
			C[(i + new_M) * K + j + new_K] = P5[idx] + P1[idx] - P3[idx] - P7[idx];
		}
	}
    delete[] P1;
    delete[] P2;
    delete[] P3;
    delete[] P4;
    delete[] P5;
    delete[] P6;
    delete[] P7;
}

void Strassen_use_GEMM_Parallel_2(const float* A, const float* B, float* C, const int M, const int N, const int K, const int strideA, const int strideB)
{
	if (M * N < 64 * 64 || N * K < 64 * 64 || M % 2 || N % 2 || K % 2)   return GEMM_Parallel_2(A, B, C, M, N, K, strideA, strideB);
	memset(C, 0, M * K * sizeof(float));

    int new_M, new_N, new_K;
    new_M = M / 2;
    new_N = N / 2;
    new_K = K / 2;

    //P1 = A11**(B21-B22), **new operator   
    float* P1 = new float[new_M * new_K];
	{
		//P1_1 = (B21-B22)
        float* P1_1 = new float[new_K * new_N];

		for (int i = new_K; i < K; i++)
		{
			for (int j = 0; j < new_N; j++)
				P1_1[(i - new_K) * new_N + j] = B[i * strideB + j] - B[i * strideB + j + new_N];
		}

		Strassen_use_GEMM_Parallel_2(A, P1_1, P1, new_M, new_N, new_K, 
            strideA, new_N);

        delete[] P1_1;
	}

    //P2 = (A11+A12)**B22
    float* P2 = new float[new_M * new_K];
	{
		//P2_0 = (A11+A12)
        float* P2_0 = new float[new_M * new_N];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P2_0[i * new_N + j] = A[i * strideA + j] + A[i * strideA + j + new_N];
		}

		Strassen_use_GEMM_Parallel_2(P2_0, B + new_K * strideB + new_N, P2, new_M, new_N, new_K,
			new_N, strideB);

         delete[] P2_0;
	}

    //P3 = (A21+A22)**B11
    float* P3 = new float[new_M * new_K];
	{
		//P3_0 = (A21+A22) 
        float* P3_0 = new float[new_M * new_N];

		for (int i = new_M; i < M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P3_0[(i - new_M) * new_N + j] = A[i * strideA + j] + A[i * strideA + j + new_N];
		}
		//M2_2 = B11
		Strassen_use_GEMM_Parallel_2(P3_0, B, P3, new_M, new_N, new_K,
			new_N, strideB);

        delete[] P3_0;
	}

    //P4 = A22**(B12-B11)
    float* P4 = new float[new_M * new_K];
	{
		//P4_1 = (B12-B11)
        float* P4_1 = new float[new_K * new_N];
	
		for (int i = 0; i < new_K; i++)
		{
			for (int j = 0; j < new_N; j++)
				P4_1[i * new_N + j] = B[i * strideB + j + new_N] - B[i * strideB + j];
		}

		Strassen_use_GEMM_Parallel_2(A + new_M * strideA + new_N, P4_1, P4, new_M, new_N, new_K,
			strideA, new_N);

        delete[] P4_1;
	}

	//P5 = (A11+A22)**(B11+B22)
    float* P5 = new float[new_M * new_K];
	{
		//P5_0= (A11+A22) 
        //P5_1 = (B11+B22)
        float* P5_0 = new float[new_M * new_N];
        float* P5_1 = new float[new_K * new_N];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P5_0[i * new_N + j] = A[i * strideA + j] + A[i * strideA + j + new_M * strideA + new_N];
		}
		
		for (int i = 0; i < new_K; i++)
		{
			for (int j = 0; j < new_N; j++)
				P5_1[i * new_N + j] = B[i * strideB + j] + B[i * strideB + j + new_K * strideB + new_N];
		}

		Strassen_use_GEMM_Parallel_2(P5_0, P5_1, P5, new_M, new_N, new_K,
            new_N, new_N);

        delete[] P5_0;
        delete[] P5_1;
	}

    //P6 = (A12-A22)**(B12+B22)
    float* P6 = new float[new_M * new_K];
	{
		//P6_0 = (A12-A22) 
        //P6_1 = (B12+B22)
        float* P6_0 = new float[new_M * new_N];
		float* P6_1 = new float[new_K * new_N];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = new_N; j < N; j++)
				P6_0[i * new_N + j - new_N] = A[i * strideA + j] - A[i * strideA + j + new_M * strideA];
		}

		for (int i = 0; i < new_K; i++)
		{
			for (int j = new_N; j < N; j++)
				P6_1[i * new_N + j - new_N] = B[i * strideB + j] + B[i * strideB + j + new_K * strideB];
		}

		Strassen_use_GEMM_Parallel_2(P6_0, P6_1, P6, new_M, new_N, new_K,
			new_N, new_N);

        delete[] P6_0;
        delete[] P6_1;
	}	

	//P7 = (A11-A21)**(B11+B21) 
    float* P7 = new float[new_M * new_K];
	{
		//P7_0 = (A11-A21) 
        //P7_1 = (B11+B21) 
        float* P7_0 = new float[new_M * new_N];
        float* P7_1 = new float[new_K * new_N];

		for (int i = 0; i < new_M; i++)
		{
			for (int j = 0; j < new_N; j++)
				P7_0[i * new_N + j] = A[i * strideA + j] - A[i * strideA + j + new_M * strideA];
		}

		for (int i = 0; i < new_K; i++)
		{
			for (int j = 0; j < new_N; j++)
				P7_1[i * new_N + j] = B[i * strideB + j + new_K * strideB] + B[i * strideB + j];
		}

		Strassen_use_GEMM_Parallel_2(P7_0, P7_1, P7, new_M, new_N, new_K,
			new_N, new_N);

        delete[] P7_0;
        delete[] P7_1;
	}


	for (int i = 0; i < new_M; i++)
	{
		for (int j = 0; j < new_K; j++)
		{
			const int idx = i * new_K + j;
			//C11 = P5 + P4 - P2 + P6
			C[i * K + j] = P5[idx] + P4[idx] - P2[idx] + P6[idx];
			//C12 = P1 + P2
			C[i * K + j + new_K] = P1[idx] + P2[idx];
			//C21 = P3 + P4
			C[(i + new_M) * K + j] = P3[idx] + P4[idx];
			//C22 = P5 + P1 - P3 - P7
			C[(i + new_M) * K + j + new_K] = P5[idx] + P1[idx] - P3[idx] - P7[idx];
		}
	}
    delete[] P1;
    delete[] P2;
    delete[] P3;
    delete[] P4;
    delete[] P5;
    delete[] P6;
    delete[] P7;
}

void check_range(const int& M, const int& N, const int& K){
   if (M > maximal_size || M < minmum_size){
       cerr << "M should be in range 512~2048\n";
       exit(-1);
   }
   else if (N > maximal_size || N < minmum_size){
       cerr << "N should be in range 512~2048\n";
       exit(-1);
   }
   else if (K > maximal_size || K < minmum_size){
       cerr << "K should be in range 512~2048\n";
       exit(-1);
   }
}

void Get_input(int& M, int& N, int& K){
    cout << "Please input M, N, K (512~2048)\n";

    cout << "M: "; cin >> M;
    cout << "N: "; cin >> N;
    cout << "K: "; cin >> K;

    check_range(M, N, K);
}

int main(int argc, char* argv[]){
    thread_count = strtol(argv[1], NULL, 10);
    omp_set_num_threads(thread_count );

    int M, N, K;
    Get_input(M, N, K);
    float *A = new float[M * N];
    float *B = new float[N * K]; 

    float *C_Strassen_use_GEMM_Serial_1 = new float[M * K];
    float *C_Strassen_use_GEMM_Serial_2 = new float[M * K];
    float *C_Strassen_use_GEMM_Parallel_2 = new float[M * K];

    float *C_MKL = new float[M * K];

    srand(time(NULL));
    Create_RandomMatrix(A, M, N);
    Create_RandomMatrix(B, N, K);
    
    cout << "Matrix A:\n";
    print_matrix(A, M, N);
    
    cout << "\nMatrix B:\n";
    print_matrix(B, N, K);

    float alpha, beta;
    alpha = 1.0; beta = 0.0;

    double start, stop;
    double time_Strassen_use_GEMM_Serial_1, 
           time_Strassen_use_GEMM_Serial_2, 
           time_Strassen_use_GEMM_Parallel_2,
           time_MKL;

    GET_TIME(start);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, K, N, alpha, A, N, B, K, beta, C_MKL, K);
    GET_TIME(stop);
    time_MKL = stop - start;

    GET_TIME(start);
    Strassen_use_GEMM_Serial_1(A, B, C_Strassen_use_GEMM_Serial_1, M, N, K, N, K);
    GET_TIME(stop);
    time_Strassen_use_GEMM_Serial_1 = stop - start; 

    transpose(B, N, K);

    GET_TIME(start);
    Strassen_use_GEMM_Serial_2(A, B, C_Strassen_use_GEMM_Serial_2, M, N, K, N, N);
    GET_TIME(stop);
    time_Strassen_use_GEMM_Serial_2 = stop - start; 

    GET_TIME(start);
    Strassen_use_GEMM_Parallel_2(A, B, C_Strassen_use_GEMM_Parallel_2, M, N, K, N, N);
    GET_TIME(stop);
    time_Strassen_use_GEMM_Parallel_2 = stop - start; 

    cout << "\nMatrix computed C by Strassen_use_GEMM_Serial_1:\n";
    print_matrix(C_Strassen_use_GEMM_Serial_1, M, K);
    
    cout << "\nMatrix computed C by Strassen_use_GEMM_Serial_2:\n";
    print_matrix(C_Strassen_use_GEMM_Serial_2, M, K);

    cout << "\nMatrix computed C by Strassen_use_GEMM_Parallel_2:\n";
    print_matrix(C_Strassen_use_GEMM_Parallel_2, M, K);

    cout << "\nMatrix computed C by MKL:\n";
    print_matrix(C_MKL, M, K);

    cout << "\nRunning time of Strassen_use_GEMM_Serial_1 is " << time_Strassen_use_GEMM_Serial_1 << " seconds\n";
    cout << "\nRunning time of Strassen_use_GEMM_Serial_2 is " << time_Strassen_use_GEMM_Serial_2 << " seconds\n";
    cout << "\nRunning time of Strassen_use_GEMM_Parallel_2 is " << time_Strassen_use_GEMM_Parallel_2 << " seconds\n";
    cout << "\nRunning time of MKL is " << time_MKL << " seconds\n";

    delete[] A;
    delete[] B;
    delete[] C_Strassen_use_GEMM_Serial_1;
    delete[] C_Strassen_use_GEMM_Serial_2;
    delete[] C_Strassen_use_GEMM_Parallel_2;
    delete[] C_MKL;
    return 0;
}