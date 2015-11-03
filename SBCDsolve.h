#include "util.h"
#include "multi.h"

class SBCDsolve{
	
	public:
	SBCDsolve(Param* param){
		
		prob = param->prob;
		lambda = param->lambda;
		C = param->C;
		
		data = &(prob->data);
		labels = &(prob->labels);
		D = prob->D;
		N = prob->N;
		K = prob->K;
		max_iter = param->max_iter;
		
	}
	
	~SBCDsolve(){
	}

	Model* solve(){
		
		//initialize alpha and v ( s.t. v = X^Talpha )
		alpha = new double*[N];
		for(int i=0;i<N;i++){
			alpha[i] = new double[K];
			for(int k=0;k<K;k++)
				alpha[i][k] = 0.0;
		}
		v = new double[D*K]; //w = prox(v);
		for(int i=0;i<D*K;i++){
			v[i] = 0.0;
		}
		/*w = new double[D*K];
		for(int i=0;i<D*K;i++){
			w[i] = 0.0;
		}*/
		
		//initialize Q_diag (Q=X*X') for the diagonal Hessian of each i-th subproblem
		Q_diag = new double[N];
		for(int i=0;i<N;i++){
			SparseVec* ins = data->at(i);
			double sq_sum = 0.0;
			for(SparseVec::iterator it=ins->begin(); it!=ins->end(); it++)
				sq_sum += it->second*it->second;
			Q_diag[i] = sq_sum;
		}
		
		//indexes for permutation of [N]
		int* index = new int[N];
		for(int i=0;i<N;i++)
			index[i] = i;
		//initialize active set out of [K] for each sample i
		int* act_k_size = new int[N];
		for(int i=0;i<N;i++)
			act_k_size[i] = K;

		int** act_k_index = new int*[N];
		for(int i=0;i<N;i++){
			act_k_index[i] = new int[K];
			for(int k=0;k<K;k++)
				act_k_index[i][k] = k;
		}
		
		//main loop
		double starttime = omp_get_wtime();
		double subsolve_time = 0.0, maintain_time = 0.0;
		double* alpha_i_new = new double[K];
		int iter = 0;
		while( iter < max_iter ){
			
			random_shuffle( index, index+N );
			for(int r=0;r<N;r++){	
				
				int i = index[r];
				int act_size = act_k_size[i];
				int* act_index = act_k_index[i];
				double* alpha_i = alpha[i];
				if( act_size < 2 )
					continue;
				
				//solve subproblem
				subsolve_time -= omp_get_wtime();
				subSolve(i, act_index, act_size, alpha_i_new);
				subsolve_time += omp_get_wtime();
				
				//maintain w = prox_{\lambda}( X^T\alpha )
				maintain_time -= omp_get_wtime();
				SparseVec* x_i = data->at(i);
				for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){

					int f_offset = it->first*K;
					double f_val = it->second;
					for(int j=0;j<act_size;j++){
						int k = act_index[j];
						int ind = f_offset+k;
						v[ ind ] += f_val*(alpha_i_new[k]-alpha_i[k]);
						//w[ ind ] = prox_l1( v[ ind ], lambda );
					}
				}
				//update alpha
				for(int j=0;j<act_size;j++){
					int k = act_index[j];
					alpha_i[k] = alpha_i_new[k];
				}
				maintain_time += omp_get_wtime();
			}
			
			if( iter % 1 == 0 )
				cerr << "." ;
			
			iter++;
		}
		double endtime = omp_get_wtime();
		cerr << endl;

		double d_obj = 0.0;
		int nSV = 0;
		int nnz_w = 0;
		int jk=0;
		for(int j=0;j<D;j++){
			for(int k=0;k<K;k++,jk++){
				if( fabs(w[jk]) > 1e-12 ){
					d_obj += w[jk]*w[jk];
					nnz_w++;
				}
			}
		}
		d_obj/=2.0;
		for(int i=0;i<N;i++){
			int yi = labels->at(i);
			for(int k=0;k<K;k++){
				if(k!=yi)
					d_obj += alpha[i][k];
				if( fabs( alpha[i][k] ) > 1e-12 )
					nSV++;
			}
		}
		cerr << "dual_obj=" << d_obj << endl;
		cerr << "nSV=" << nSV << " (NK=" << N*K << ")"<< endl;
		cerr << "nnz_w=" << nnz_w << " (DK=" << D*K << ")" << endl;
		cerr << "train time=" << endtime-starttime << endl;
		cerr << "subsolve time=" << subsolve_time << endl;
		cerr << "maintain time=" << maintain_time << endl;

		//delete algorithm-specific variables
		delete[] alpha_i_new;
		delete[] act_k_size;
		for(int i=0;i<N;i++)
			delete[] act_k_index[i];
		delete[] act_k_index;
		
		for(int i=0;i<N;i++)
			delete[] alpha[i];
		delete[] alpha;
		delete[] Q_diag;
		
		return new Model(prob, w);
	}
	
	void subSolve(int i, int* act_k_index, int act_k_size, double* alpha_i_new){
		
		grad = new double[act_k_size];
		Dk = new double[act_k_size];
		
		int yi = labels->at(i);
		SparseVec* x_i = data->at(i);
		double Qii = Q_diag[i];
		double* alpha_i = alpha[i];
		//compute gradient of each k
		for(int j=0;j<act_k_size;j++){
			int k = act_k_index[j];
			if( k!=yi )
				grad[j] = 1.0 - Qii*alpha_i[k];
			else
				grad[j] = - Qii*alpha_i[k];
		}
		for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
			int fea_offset = it->first*K;
			double fea_val = it->second;
			for(int j=0;j<act_k_size;j++){
				int k = act_k_index[j];
				double vjk = v[fea_offset+k];
				if( fabs(vjk) > lambda ){
					if( vjk > 0 )
						grad[j] += (vjk-lambda)*fea_val;
					else
						grad[j] += (vjk+lambda)*fea_val;
				}
			}
		}
		for(int j=0;j<act_k_size;j++){
			int k = act_k_index[j];
			if( k!=yi )
				Dk[j] = grad[j];
			else
				Dk[j] = grad[j] + Qii*C;
		}
		
		//sort according to D_k = grad_k + Qii*((k==yi)?C:0)
		sort( Dk, Dk+act_k_size, greater<double>() );
		
		//compute beta by traversing k in descending order of D_k
		double beta = Dk[0] - Qii*C;
		int r;
		for(r=1;r<act_k_size && beta<r*Dk[r];r++){
			beta += Dk[r];
		}
		beta = beta / r;
		
		//update alpha
		for(int j=0;j<act_k_size;j++){
			int k = act_k_index[j];
			alpha_i_new[k] = min( (k!=yi)?0.0:C, (beta-grad[j])/Qii );
		}
		
		delete[] grad;
		delete[] Dk;
	}

	private:
	Problem* prob;
	double lambda;
	double C;
	vector<SparseVec*>* data ;
	vector<int>* labels ;
	int D; 
	int N;
	int K;
	double* Q_diag;
	double** alpha;
	double* v;
	double* w;
	
	int max_iter;
	double* grad;
	double* Dk;
};
