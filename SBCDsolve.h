#include "util.h"
#include "multi.h"
#include "newHash.h"
#include <cassert>
class SBCDsolve{
	
	public:
	SBCDsolve(Param* param){
		
		prob = param->train;
		heldoutEval = param->heldoutEval;
		early_terminate = param->early_terminate;
		lambda = param->lambda;
		C = param->C;
		
		data = &(prob->data);
		labels = &(prob->labels);
		D = prob->D;
		N = prob->N;
		K = prob->K;
		max_iter = param->max_iter;
		#ifdef USING_HASHVEC
		hashfunc = new HashClass(K);
		hashindices = hashfunc->hashindices;
		#endif
	}
	
	~SBCDsolve(){
		for(int i=0;i<N;i++)
			delete[] alpha[i];
		delete[] alpha;
		for(int j=0;j<D;j++)
			delete[] v[j];
		delete[] v;
	}
	
	Model* solve(){
		
		//initialize alpha and v ( s.t. v = X^Talpha )
		#ifdef USING_HASHVEC
		alpha = new pair<int, Float>*[N];
		size_alpha = new int[N];
		util_alpha = new int[N];
		memset(util_alpha, 0, N*sizeof(int));
		for (int i = 0; i < N; i++){
			size_alpha[i] = 1;
			while(size_alpha[i] < K){
				size_alpha[i] = size_alpha[i] << 1;
			}
			alpha[i] = new pair<int, Float>[size_alpha[i]];
			for (int k = 0; k < size_alpha[i]; k++){
				alpha[i][k] = make_pair(-1, 0.0);
			}
		}
		v = new pair<int, Float>*[D];
		size_v = new int[D];
		util_v = new int[D];
		memset(util_v, 0, D*sizeof(int));
		for (int j = 0; j < D; j++){
			size_v[j] = INIT_SIZE;
			v[j] = new pair<int, Float>[size_v[j]];
			for(int k = 0; k < size_v[j]; k++){
				v[j][k] = make_pair(-1, 0.0);
			}
		}
		#else
		alpha = new Float*[N];
		for(int i=0;i<N;i++){
			alpha[i] = new Float[K];
			for(int k=0;k<K;k++)
				alpha[i][k] = 0.0;
		}
		v = new pair<Float, Float>*[D]; //w = prox(v);
		for(int j=0;j<D;j++){
			v[j] = new pair<Float, Float>[K];
			for(int k=0;k<K;k++)
				v[j][k] = make_pair(0.0, 0.0);
		}
		#endif
		//initialize Q_diag (Q=X*X') for the diagonal Hessian of each i-th subproblem
		Q_diag = new Float[N];
		for(int i=0;i<N;i++){
			SparseVec* ins = data->at(i);
			Float sq_sum = 0.0;
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
			for(int k=0;k<K;k++){
				act_k_index[i][k] = k;
			}
		}
		//main loop
		double max_heldout_test_acc = 0.0;
		int terminate_countdown = 0;
		double starttime = omp_get_wtime();
		double subsolve_time = 0.0, maintain_time = 0.0;
		Float* alpha_i_new = new Float[K];
		int iter = 0;
		while( iter < max_iter ){
			random_shuffle( index, index+N );
			for(int r=0;r<N;r++){
				
				int i = index[r];
				int act_size = act_k_size[i];
				int* act_index = act_k_index[i];
				#ifdef USING_HASHVEC
				pair<int, Float>* alpha_i = alpha[i];
				int size_alphai = size_alpha[i];
				int size_alphai0 = size_alphai - 1;
				int index_alpha = 0, index_v = 0;
				#else
				Float* alpha_i = alpha[i];
				#endif
				if( act_size < 2 )
					continue;
				
				//solve subproblem
				subsolve_time -= omp_get_wtime();
				subSolve(i, act_index, act_size, alpha_i_new);
				subsolve_time += omp_get_wtime();
				
				//maintain w = prox_{\lambda}( X^T\alpha )
				maintain_time -= omp_get_wtime();
				SparseVec* x_i = data->at(i);
				#ifdef USING_HASHVEC
				Float* alpha_i_k = new Float[act_size];
				for(int j = 0; j < act_size; j++){
					int act_indexj = act_index[j];
					find_index(alpha_i, index_alpha, act_indexj, size_alphai0, hashindices);
					alpha_i_k[j] = alpha_i_new[act_indexj] - alpha_i[index_alpha].second; 
				}
				#endif
				for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){

					int J = it->first;
					Float f_val = it->second;
					#ifdef USING_HASHVEC
					pair<int, Float>* vj = v[J];
					int size_vj = size_v[J];
					int util_vj = util_v[J];
					int size_vj0 = size_vj - 1;
					for(int j = 0; j < act_size; j++){
						int act_indexj = act_index[j];
						find_index(vj, index_v, act_indexj, size_vj0, hashindices);
						vj[index_v].second += f_val*alpha_i_k[j];
						if (vj[index_v].first == -1){
							vj[index_v].first = act_indexj;
							if ((++util_vj) > size_vj * UPPER_UTIL_RATE){
								//resize here
								resize(vj, v[J], size_v[J], size_vj, size_vj0, util_vj, hashindices);
							}
						}
					}
					util_v[J] = util_vj;
					#else
					pair<Float, Float>* vj = v[J];
					for(int j=0;j<act_size;j++){
						int k = act_index[j];
						vj[k].first += f_val*(alpha_i_new[k]-alpha_i[k]);
						vj[k].second = prox_l1(vj[k].first, lambda);
					}
					#endif
				}
				//update alpha
				#ifdef USING_HASHVEC
				for(int j=0;j<act_size;j++){
					int act_indexj = act_index[j];
					find_index(alpha_i, index_alpha, act_indexj, size_alphai0, hashindices);
					
					if (alpha_i[index_alpha].first == -1 && (alpha_i_new[act_indexj] != 0.0)){
						alpha_i[index_alpha].first = act_indexj;
					};
					alpha_i[index_alpha].second = alpha_i_new[act_indexj];
				}
				if (act_size > size_alphai * UPPER_UTIL_RATE){
					//resize here
					resize(alpha_i, alpha[i], size_alpha[i], size_alphai, size_alphai0, act_size, hashindices);
				}
				delete[] alpha_i_k;
				#else
				for(int j=0;j<act_size;j++){
					int k = act_index[j];
					alpha_i[k] = alpha_i_new[k];
				}
				#endif
			}
			
			maintain_time += omp_get_wtime();
			if( iter % 1 == 0 )
				cerr << "." ;
			
			if (heldoutEval != NULL){
				#ifdef USING_HASHVEC
				Float heldout_test_acc = heldoutEval->calcAcc(v, size_v, hashindices, lambda);
				#else
				Float heldout_test_acc = heldoutEval->calcAcc(v);
				#endif	
				cerr << "heldout Acc=" << heldout_test_acc << " ";
				if ( heldout_test_acc > max_heldout_test_acc){
					max_heldout_test_acc = heldout_test_acc;
					terminate_countdown = 0;
				} else {
					cerr << "(" << (++terminate_countdown) << "/" << early_terminate << ")";
					if (terminate_countdown == early_terminate)
						break;
				}
				cerr << endl;
			}
			iter++;
		}
		double endtime = omp_get_wtime();
		cerr << endl;
		
		//convert v into w
		#ifdef USING_HASHVEC
		w = new pair<int, Float>*[D];
		size_w = new int[D];
		nnz_index = new vector<int>[D];
		for (int j = 0; j < D; j++){
			size_w[j] = 2;
			w[j] = new pair<int, Float>[size_w[j]];
			for (int tt = 0; tt < size_w[j]; tt++){
				w[j][tt] = make_pair(-1, 0.0);
			}
			nnz_index[j].clear();
		}
		for (int j = 0; j < D; j++){
			pair<int, Float>* wj = w[j];
			int size_wj = size_w[j];
			int size_wj0 = size_wj - 1;
			for (int k = 0; k < K; k++){
				int index_v = 0;
				find_index(v[j], index_v, k, size_v[j] - 1, hashindices);
				Float wjk = v[j][index_v].second;
				if ( fabs(wjk) > 1e-12 ){
					int index_w = 0;
					find_index(wj, index_w, k, size_wj - 1, hashindices);
					wj[index_w].second = wjk;
					if (wj[index_w].first == -1){
						wj[index_w].first = k;
						nnz_index[j].push_back(k);
						if (size_wj* UPPER_UTIL_RATE < nnz_index[j].size()){
							int util = nnz_index[j].size();
							resize(wj, w[j], size_wj, size_w[j], size_wj0, util, hashindices);
						}
					}
				}
			}
		}
		#else
		w = new Float*[D];
		nnz_index = new vector<int>[D];
		for (int j = 0; j < D; j++){
			w[j] = new Float[K];
			for (int k = 0; k < K; k++){
				w[j][k] = 0.0;
			}
			nnz_index[j].clear();
		}
		for (int j = 0; j < D; j++){
			for (int k = 0; k < K; k++){
				Float wjk = v[j][k].second;
				if (fabs(wjk) > 1e-12){
					w[j][k] = wjk;
					nnz_index[j].push_back(k);
				}
			}
		}
		
		#endif
		
		Float d_obj = 0.0;
		int nSV = 0;
		int nnz_w = 0;
		double w_1norm=0.0;
		for(int j=0;j<D;j++){
			for(vector<int>::iterator it=nnz_index[j].begin(); it!=nnz_index[j].end(); it++){
				int k = *it;
				#ifdef USING_HASHVEC
				int index_w = 0;
				find_index(w[j], index_w, k, size_w[j]-1, hashindices);
				Float wjk = w[j][index_w].second;
				#else
				Float wjk = w[j][k];
				#endif
				d_obj += wjk*wjk;
				w_1norm += fabs(wjk);
				
			}
			nnz_w += nnz_index[j].size();
		}
		d_obj/=2.0;
		for(int i=0;i<N;i++){
			Labels* yi = &(labels->at(i));
			
			for(int k=0;k<K;k++){
				
				#ifdef USING_HASHVEC
				int index_alpha = 0;
				find_index(alpha[i], index_alpha, k, size_alpha[i] - 1, hashindices);
				if(find(yi->begin(), yi->end(), k) ==yi->end())
					d_obj += alpha[i][index_alpha].second;
				if( fabs( alpha[i][index_alpha].second ) > 1e-12 )
					nSV++;
				#else
				if(find(yi->begin(), yi->end(), k) ==yi->end())
					d_obj += alpha[i][k];
				if( fabs( alpha[i][k] ) > 1e-12 )
					nSV++;
				#endif
			}
		}
		cerr << "dual_obj=" << d_obj << endl;
		cerr << "nSV=" << nSV << " (NK=" << N*K << ")"<< endl;
		cerr << "nnz_w=" << nnz_w << " (DK=" << D*K << ")" << endl;
		cerr << "w_1norm=" << w_1norm << endl;
		cerr << "train time=" << endtime-starttime << endl;
		cerr << "subsolve time=" << subsolve_time << endl;
		cerr << "maintain time=" << maintain_time << endl;
		
		//delete algorithm-specific variables
		delete[] alpha_i_new;
		delete[] act_k_size;
		for(int i=0;i<N;i++)
			delete[] act_k_index[i];
		delete[] act_k_index;
		
		delete[] Q_diag;
		
		#ifdef USING_HASHVEC
		return new Model(prob, nnz_index, w, size_w, hashindices);
		#else
		return new Model(prob, nnz_index, w);
		#endif
	}
	
	void subSolve(int I, int* act_k_index, int act_k_size, Float* alpha_i_new){
		Labels* yi = &(labels->at(I));
		int m = yi->size(), n = act_k_size - m;
		Float* b = new Float[n];
		Float* c = new Float[m];
		int* act_index_b = new int[n];
		int* act_index_c = new int[m];
		
		SparseVec* x_i = data->at(I);
		Float A = Q_diag[I];
		#ifdef USING_HASHVEC
		pair<int, Float>* alpha_i = alpha[I];
		int size_alphai = size_alpha[I];
		int size_alphai0 = size_alphai - 1;
		int index_alpha = 0;
		#else
		Float* alpha_i = alpha[I];
		#endif
		int i = 0, j = 0;
		for(int k=0;k<m+n;k++){
			int p = act_k_index[k];
			#ifdef USING_HASHVEC
			find_index(alpha_i, index_alpha, p, size_alphai0, hashindices);
			
			if( find(yi->begin(), yi->end(), p) == yi->end() ){
				b[i] = 1.0 - A*alpha_i[index_alpha].second;
				act_index_b[i++] = p;
			}else{
                                c[j] = A*alpha_i[index_alpha].second;
				act_index_c[j++] = p;
			}
			#else
			if( find(yi->begin(), yi->end(), p) == yi->end() ){
				b[i] = 1.0 - A*alpha_i[p];
				act_index_b[i++] = p;
			}else{
				c[j] = A*alpha_i[p];
				act_index_c[j++] = p;
			}
			#endif
		}

		for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
			int fea_ind = it->first;
			Float fea_val = it->second;
			#ifdef USING_HASHVEC
			pair<int, Float>* vj = v[fea_ind];
			int size_vj = size_v[fea_ind];
			int size_vj0 = size_vj - 1;
			int index_v = 0;
			#else
			pair<Float, Float>* vj = v[fea_ind];
			#endif
			for(int i = 0; i < n; i++){
				int k = act_index_b[i];
				#ifdef USING_HASHVEC
				find_index(vj, index_v, k, size_vj0, hashindices);
				Float vjk = vj[index_v].second;
				if (fabs(vjk) > lambda){
					if (vjk > 0)
						b[i] += (vjk-lambda)*fea_val;
					else
						b[i] += (vjk+lambda)*fea_val;
				}
				#else
				Float wjk = vj[k].second;
				b[i] += wjk*fea_val;
				#endif
			}
			for(int j = 0; j < m; j++){
				int k = act_index_c[j];
				#ifdef USING_HASHVEC
				find_index(vj, index_v, k, size_vj0, hashindices);
				Float vjk = vj[index_v].second;	
				if( fabs(vjk) > lambda ){
					if( vjk > 0 )
						c[j] -= (vjk-lambda)*fea_val;
					else
						c[j] -= (vjk+lambda)*fea_val;
				}
				#else
				c[j] -= vj[k].second*fea_val;
				#endif
			}
		}

		for (int i = 0; i < n; i++){
			b[i] /= A;
		}
		for (int j = 0; j < m; j++){
			c[j] /= A;
		}
		
		Float* x = new Float[n];
		Float* y = new Float[m];
		solve_bi_simplex(n, m, b, c, C, x, y);
		
		for(int i = 0; i < n; i++){
			int k = act_index_b[i];
			alpha_i_new[k] = -x[i]; 
		}
		for(int j = 0; j < m; j++){
                        int k = act_index_c[j];
                        alpha_i_new[k] = y[j]; 
                }
		
		delete[] x; delete[] y;	
		delete[] b; delete[] c;
		delete[] act_index_b; delete[] act_index_c;
	}

	private:
	Problem* prob;
	HeldoutEval* heldoutEval;
	Float lambda;
	Float C;
	vector<SparseVec*>* data ;
	vector<Labels>* labels ;
	int D; 
	int N;
	int K;
	Float* Q_diag;
	
	//heldout options
	int early_terminate;
	
	vector<int>* nnz_index;
	#ifdef USING_HASHVEC
	pair<int, Float>** w;
	int* size_w;
	pair<int, Float>** v;
	pair<int, Float>** alpha;
	HashClass* hashfunc;
	int* hashindices;
	int* size_v;
	int* size_alpha;
	int* util_v;
	int* util_alpha;
	#else
	Float** w;
	Float** alpha;
	pair<Float, Float>** v;
	#endif
		
	int max_iter;
	Float* grad;
	Float* Dk;
};
