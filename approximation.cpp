
#include <iostream>
#include "approximation.h"
#include <math.h>
#include <ctime>
#include <Eigen/Core>
// AVX instructions
#include "immintrin.h"
using Eigen::MatrixXf;
using namespace std;

// approximation of C = A*B
MatrixXf prod_approx(MatrixXf A,MatrixXf B)
{       
    // cout<<"A.rows()"<<A.rows()<<endl;
    // cout<<"A.cols()"<<A.cols()<<endl;
    // cout<<"A.colwise().mean().rows()"<<A.colwise().mean().rows()<<endl;
    // cout<<"A.colwise().mean().cols()"<<A.colwise().mean().cols()<<endl;
    // cout<<"A.rowwise().mean().rows()"<<A.rowwise().mean().rows()<<endl;
    // cout<<"A.rowwise().mean().cols()"<<A.rowwise().mean().cols()<<endl;
    // cout<<"B.rows()"<<B.rows()<<endl;
    // cout<<"B.cols()"<<B.cols()<<endl;
    // cout<<"B.colwise().mean().rows()"<<B.colwise().mean().rows()<<endl;
    // cout<<"B.colwise().mean().cols()"<<B.colwise().mean().cols()<<endl;
    // cout<<"B.rowwise().mean().rows()"<<B.rowwise().mean().rows()<<endl;
    // cout<<"B.rowwise().mean().cols()"<<B.rowwise().mean().cols()<<endl;
    
    // MatrixXf C(A.rows(),B.cols());
    // C.setZero();
    
    // MatrixXf B_transpose = B.transpose();
    
    // for (int k=0;k<A.rows();k+=1)
    // {   
        // for(int j=0;j<B.cols();j+=1)
        // {   
            // for(int i=0;i<B.rows();i+=1)
            // {             
                // C(k,j) += A(k,i)*B_transpose(j,i);
            // }
        // }
    // }
    
    // block matrix multiplication
    // in order to optimize memory accesses, I am using loop tiling and sequential accesses
    // https://en.wikipedia.org/wiki/Loop_tiling
    // https://en.wikipedia.org/wiki/Locality_of_reference
    

    int I = (int)A.rows();
    int J = (int)B.cols();
    int K = (int)B.rows();
    
    int b0 = 64;
    MatrixXf C(I,J);
    C.setZero();
    
    float sum;
    int i, j, k, k0, j0;
    int J0,K0;
    
    MatrixXf B_transpose = B.transpose();
    
    // omp_set_num_threads(4); 
    // #pragma omp parallel shared(C) private(k0,j0,J0,K0,i,j,k,sum) num_threads(2)
    // {
    // #pragma omp for
    for (k0=0;k0<K;k0+=b0)
    {   
        for(j0=0;j0<J;j0+=b0)
        {   
            J0 = min(j0+b0,J);
            K0 = min(k0+b0,K);

            // The i0 an i for loops are merged
            for(i=0;i<I;i+=1)
            {   
                // oddly, using min makes the code slower
                // maybe it prevents some vetorization optim?
                for(j=j0;j<J0;j++)
                // for(j=j0;j<min(j0+b0,J);j++)
                // for(j=j0;j<j0+b0;j++)
                {
                    sum = C(i,j);
                    for(k=k0;k<K0;k++)
                    // for(k=k0;k<min(k0+b0,K);k++)
                    // for(k=k0;k<k0+b0;k++)
                    {
                        sum += A(i,k)*B_transpose(j,k);
                    }
                    C(i,j) = sum;
                    
                }
            }
        }
    }
    // }
    
    // for (int k=0;k<A.rows();k+=1)
    // {  
        // for(int j=0;j<B.cols();j+=1)
        // { 
            // cout<<"C(k,j) = "<<C(k,j)<<endl;
        // }
    // }
    // cin.get();
    
    /*
    // correct but slow approx
    MatrixXf C;
    C.resize(A.rows(),B.cols());
    // float scaler = 1./4.;
    float scaler = 1./(4*A.cols());
    
    MatrixXf B_transpose = B.transpose();
    
    for (int k=0; k<A.rows(); ++k)
    {   
        for(int j=0;j<B.cols();j++)
        {            
            float sumA = 0;
            float sumB = 0;
            for(int i=0;i<B.rows();i++)
            {
                sumA += abs(A(k,i)+B_transpose(j,i));
                sumB += abs(A(k,i)-B_transpose(j,i));
                
                // sumA += pow(A(k,i)+B_transpose(j,i),2);
                // sumB += pow(A(k,i)-B_transpose(j,i),2);
                
            }
            C(k,j)= scaler*(pow(sumA,2)-pow(sumB,2));
            
            // C(k,j)= scaler*(sumA-sumB);
        }
    }
    
    // clock_t start_time;
    // float elapsed_time;
    
    // start_time = clock();
    
    // MatrixXf sumA = A.rowwise().sum();
    
    
    MatrixXf sumA;
    sumA.resize(1,A.rows());
    
    for(int k=0;k<A.rows();k++)
	{
        sumA(0,k) = 0;
        for (int i = 0; i <A.cols(); ++i)
		{
            sumA(0,k) += A(k,i);
        }
    } 
    
    MatrixXf sumB;
    sumB.resize(1,B.cols());
    
    for (int j=0; j<B.cols(); ++j)
    {   
        sumB(0,j) = 0;
        for(int i=0;i<B.rows();i++)
        {
            // non sequential access to B...
            sumB(0,j) += B(i,j);
        }
    } 
    
    MatrixXf C;
    C.resize(A.rows(),B.cols());
    // float scaler = 1./(4*A.cols());
    float scaler = 1./A.cols();
    
    for (int k=0; k<A.rows(); ++k)
    {   
        for(int j=0;j<B.cols();j++)
        {
            // C(k,j)= scaler * (pow((sumA(0,k)+sumB(0,j)),2)-pow((sumA(0,k)-sumB(0,j)),2));
            C(k,j)= scaler*sumA(0,k)*sumB(0,j);
        }
    }
    
    */
    
    return C;
    
    // for(int i=0;i<1000;i++) sumA = A.rowwise().mean();
    // elapsed_time = float(clock()-start_time)/ (CLOCKS_PER_SEC*Eigen::nbThreads());
    // cout <<endl<<"	A mean time =  " << elapsed_time;
    
    // start_time = clock();
    
    // MatrixXf sumB = B.colwise().sum();
    
    // for(int i=0;i<1000;i++) sumB = B.colwise().mean();
    // elapsed_time = float(clock()-start_time)/ (CLOCKS_PER_SEC*Eigen::nbThreads());
    // cout <<endl<<"	B mean time = " << elapsed_time;
    
    // start_time = clock();
    
    // MatrixXf C = sumA *sumB;
    
    // for(int i=0;i<1000;i++) C = sumA *sumB;
    // elapsed_time = float(clock()-start_time)/ (CLOCKS_PER_SEC*Eigen::nbThreads());
    // cout <<endl<<"	C mean time = " << elapsed_time;
    
    return C;
    
    // cin.get();
    // return (A.rowwise().mean())*(B.colwise().mean());
}