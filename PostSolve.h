#include "util.h"
#include "multi.h"

class PostSolve{
	
	public:
	#ifdef USING_HASHVEC
	PostSolve(Param* param, vector<int>* _w_hash_nnz_index, pair<int, Float>** _w, int* _size_w, vector<pair<int, Float>>* _act_k_index, int*& _hashindices){
	#else
	PostSolve(Param* param, vector<int>* _w_hash_nnz_index, Float** _w, vector<pair<int, Float>>* _act_k_index){
	#endif
		
		double construct_time = -omp_get_wtime();
		
		prob = param->train;
		C = param->C;
		lambda = param->lambda;	
	
		vector<SparseVec*>* data = &(prob->data);
		labels = &(prob->labels);
		D = prob->D;
		N = prob->N;
		K = prob->K;
		max_iter = param->post_solve_iter;
		int maxDK = D;
		if (maxDK < K) 
			maxDK = K;
		hashfunc = new HashClass(maxDK);
		hashindices = hashfunc->hashindices;
		
		//initialize alpha and v
		act_k_index = new vector<pair<int, Float>>[N];
		for(int i = 0; i < N; i++){
			act_k_index[i] = _act_k_index[i];
		}
		#ifdef USING_HASHVEC
		v = new pair<int, Float>*[K];
		size_v = new int[K];
		util_v = new int[K];
		for(int k = 0; k < K; k++){
			size_v[k] = 1;
			util_v[k] = 0;
		}
		for(int j = 0; j < D; j++){
			for (int it = 0; it < _size_w[j]; it++){
				int k = _w[j][it].first;
				if (k == -1)
					continue;
				if (fabs(_w[j][it].second) <= 1e-12)
					continue;
				util_v[k]++;
				//count_v++;
				if (size_v[k]*UPPER_UTIL_RATE < util_v[k])
					size_v[k]*= 2;
			}
		}
		for(int k = 0; k < K; k++){
			v[k] = new pair<int, Float>[size_v[k]];
			for (int j = 0; j < size_v[k]; j++)
				v[k][j] = make_pair(-1, 0.0);
		}
		for (int j = 0; j < D; j++){
			for (int it = 0; it < _size_w[j]; it++){
				int k = _w[j][it].first;
				if (k == -1)
					continue;
				Float vkj = _w[j][it].second;
				if (fabs(vkj) <= 1e-12)
					continue;
				if (vkj > 0)
					vkj += param->lambda;
				else
					vkj -= param->lambda;
				//insert (j, _v[j][it]) to v[k]
				int index_v = 0;
				find_index(v[k], index_v, j, size_v[k]-1, hashindices);
				v[k][index_v] = make_pair(j, vkj);
			}
		}
		#else
		v = new Float*[K]; //w = prox(v);
		for(int k=0;k<K;k++){
			v[k] = new Float[D];
			memset(v[k], 0, sizeof(Float)*D);
		}
		for(int j=0;j<D;j++)
			for(vector<int>::iterator it=_w_hash_nnz_index[j].begin(); it!=_w_hash_nnz_index[j].end(); it++){
				int k = *it;
				Float vkj = _w[j][k];
				if (vkj > 0)
					vkj += param->lambda;
				else
					vkj -= param->lambda;
				v[k][j] = vkj;
			}
		#endif

		// construct data_per_class
		data_per_class = new vector<SparseVec*>[N];
			
		#ifdef USING_HASHVEC
		for(int i=0;i<N;i++){
			SparseVec* xi = data->at(i);
			vector<SparseVec*>* data_per_class_i = &(data_per_class[i]);
			vector<pair<int, Float>>* act_k_i = &(act_k_index[i]);
			for(vector<pair<int, Float>>::iterator it=act_k_i->begin(); it!=act_k_i->end(); it++){
				int k = it->first;
				SparseVec* data_per_class_i_k = new SparseVec();
				//loop over j s.t. x[i][j] != 0
				for(SparseVec::iterator it2=xi->begin(); it2!=xi->end(); it2++){
					//need w[j][k] != 0
					int index_w = 0;
					int j = it2->first;
					find_index(_w[j], index_w, k, _size_w[j]-1, _hashindices);
					if( _w[j][index_w].second != 0.0 )
						data_per_class_i_k->push_back(make_pair(j, it2->second));
				}
				data_per_class_i->push_back(data_per_class_i_k);
			}
		}

		#else
		for(int i=0;i<N;i++){
			SparseVec* xi = data->at(i);
			vector<SparseVec*>* data_per_class_i = &(data_per_class[i]);
			vector<pair<int, Float>>* act_k_i = &(act_k_index[i]);
			for(vector<pair<int, Float>>::iterator it=act_k_i->begin(); it!=act_k_i->end(); it++){
				int k = it->first;
				SparseVec* data_per_class_i_k = new SparseVec();
				for(SparseVec::iterator it2=xi->begin(); it2!=xi->end(); it2++){
					if( _w[ it2->first ][k] != 0.0 )
						data_per_class_i_k->push_back(make_pair(it2->first, it2->second));
				}
				data_per_class_i->push_back(data_per_class_i_k);
			}
		}
		#endif
		
		construct_time += omp_get_wtime();
		cerr << "construct_time=" << construct_time << endl;

		//initialize Q_diag (Q=X*X') for the diagonal Hessian of each i-th subproblem
		Q_diag = new Float[N];
		for(int i=0;i<N;i++){
			SparseVec* ins = data->at(i);
			Float sq_sum = 0.0;
			for(SparseVec::iterator it=ins->begin(); it!=ins->end(); it++)
				sq_sum += it->second*it->second;
			Q_diag[i] = sq_sum;
		}
	}
	
	~PostSolve(){
		for(int i=0;i<N;i++){
			for (vector<SparseVec*>::iterator it = data_per_class[i].begin(); it != data_per_class[i].end(); it++){
				SparseVec* data_per_class_i_k = *it;
				data_per_class_i_k->clear();
			}
			data_per_class[i].clear();
		}
		delete[] data_per_class;
		delete[] act_k_index;
		
		delete[] Q_diag;
		for(int k=0;k<K;k++)
			delete[] v[k];
		delete[] v;
	}
	
	Model* solve(){
		
		//indexes for permutation of [N]
		int* index = new int[N];
		for(int i=0;i<N;i++)
			index[i] = i;
		//main loop
		double starttime = omp_get_wtime();
		double subsolve_time = 0.0, maintain_time = 0.0;
		Float* alpha_i_new = new Float[K];
		int iter = 0;
		while( iter < max_iter ){
			random_shuffle( index, index+N );
			for(int r=0;r<N;r++){
				
				int i = index[r];
				
				if( act_k_index[i].size() < 2 )
					continue;
				
				//solve subproblem
				subsolve_time -= omp_get_wtime();
				subSolve(i, act_k_index[i], alpha_i_new);
				subsolve_time += omp_get_wtime();
				
				//maintain v = X^T\alpha
				maintain_time -= omp_get_wtime();
				vector<SparseVec*>* x_i_per_class = &(data_per_class[i]);
				vector<SparseVec*>::iterator data_it = x_i_per_class->begin();
				for(vector<pair<int, Float>>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
					
					int k= it->first;
					SparseVec* x_i = *(data_it++);
					Float alpha_diff = alpha_i_new[k] - it->second;
					#ifdef USING_HASHVEC
					int size_vk0 = size_v[k] - 1;
					int size_vk = size_v[k];
					pair<int, Float>* vk = v[k];
					for(SparseVec::iterator it2=x_i->begin() ;it2!=x_i->end(); it2++){
						int index_v = 0, j = it2->first;
						find_index(vk, index_v, j, size_vk0, hashindices);
						vk[index_v].second += it2->second*(alpha_diff);
						if (vk[index_v].first == -1){
							vk[index_v].first = j;
							if ((++util_v[k]) > size_v[k]*UPPER_UTIL_RATE){
								resize(vk, v[k], size_vk, size_v[k], size_vk0, util_v[k], hashindices);	
							}
						}
					}
					#else
					Float* vk = v[k];
					for(SparseVec::iterator it2=x_i->begin() ;it2!=x_i->end(); it2++){
						vk[it2->first] += it2->second*(alpha_diff);
					}
					#endif
				}
				for(vector<pair<int, Float>>::iterator it = act_k_index[i].begin(); it != act_k_index[i].end(); it++){
					it->second = alpha_i_new[it->first];
				}
				maintain_time += omp_get_wtime();
			}
			
			if( iter % 1 == 0 )
				cerr << "." ;
			
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
		for (int k = 0; k < K; k++){
			for (int it = 0; it < size_v[k]; it++){
				int j = v[k][it].first;
				if (j == -1)
					continue;
				pair<int, Float>* wj = w[j];
				int size_wj = size_w[j];
				if (fabs(v[k][it].second) > 1e-12 ){
					int index_w = 0;
					find_index(wj, index_w, k, size_wj - 1, hashindices); 	
					wj[index_w].second = v[k][it].second;
					if (wj[index_w].first == -1){
						wj[index_w].first = k;
						nnz_index[j].push_back(k);
						if (size_wj* UPPER_UTIL_RATE < nnz_index[j].size()){
							int util = nnz_index[j].size();
							int size_wj0 = size_wj - 1;
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
		for (int k = 0; k < K; k++){
			for (int j = 0; j < D; j++){
				Float wjk = v[k][j];
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
				wjk = prox_l1(wjk, lambda);
				d_obj += wjk*wjk;
				w_1norm += fabs(wjk);
			}
			nnz_w += nnz_index[j].size();
		}
		d_obj/=2.0;
		for(int i=0;i<N;i++){
			Labels* yi = &(labels->at(i));
			for (vector<pair<int, Float>>::iterator it = act_k_index[i].begin(); it != act_k_index[i].end(); it++){
				int k = it->first;
				
				if(find(yi->begin(), yi->end(), k) ==yi->end())
					d_obj += it->second;
				if( fabs( it->second ) > 1e-12 )
					nSV++;
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
		delete[] index;
		
		#ifdef USING_HASHVEC
		return new Model(prob, nnz_index, w, size_w, hashindices);
		#else
		return new Model(prob, nnz_index, w);
		#endif
	}
	
	void subSolve(int I, vector<pair<int, Float>>& act_k_index, Float* alpha_i_new){
		
		Labels* yi = &(labels->at(I));
		int act_k_size = act_k_index.size();
		
		//number of indices in (yi) and (act_set - yi), respectively
		int j = 0, i = 0;
		int m = yi->size(), n = act_k_size - m;
		
		Float* b = new Float[n+m];
		Float* c = new Float[m+n];
		int* act_index_b = new int[n+m];
		int* act_index_c = new int[m+n];
		vector<SparseVec*>* x_i_per_class = &(data_per_class[I]);
		
		Float A = Q_diag[I];

		vector<SparseVec*>::iterator data_it = x_i_per_class->begin();
		for(vector<pair<int, Float>>::iterator it = act_k_index.begin(); it != act_k_index.end(); it++){
			int k = it->first;
			Float alpha_ik = it->second;
                        if( find(yi->begin(), yi->end(), k) == yi->end() ){
                                b[i] = 1.0 - A*alpha_ik;
				act_index_b[i] = k;
				SparseVec* x_i = *(data_it++);
				#ifdef USING_HASHVEC
				int size_vk0 = size_v[k] - 1;
				pair<int, Float>* vk = v[k];
				for(SparseVec::iterator it2 = x_i->begin(); it2!=x_i->end(); it2++){
					int index_v = 0, J = it2->first;
					find_index(vk, index_v, J, size_vk0, hashindices);
					b[i] += vk[index_v].second*it2->second;
				}
				#else
				Float* vk = v[k];
				for(SparseVec::iterator it2=x_i->begin(); it2!=x_i->end(); it2++){
					b[i] += vk[it2->first]*it2->second;
				}
				#endif
				i++;
                        }else{
                                c[j] = A*alpha_ik;
				act_index_c[j] = k;
				SparseVec* x_i = *(data_it++);
				#ifdef USING_HASHVEC
				int size_vk0 = size_v[k] - 1;
				pair<int, Float>* vk = v[k];
				for(SparseVec::iterator it2 = x_i->begin(); it2!=x_i->end(); it2++){
					int index_v = 0, J = it2->first;
					find_index(vk, index_v, J, size_vk0, hashindices);
					c[j] -= vk[index_v].second*it2->second;
				}
				#else
				Float* vk = v[k];
				for(SparseVec::iterator it2=x_i->begin(); it2!=x_i->end(); it2++)
					c[j] -= vk[it2->first]*it2->second;
				#endif
				j++;
                        }
		}
		n = i;
		m = j;
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
	Float C, lambda;
	
	vector<SparseVec*>* data_per_class;
	vector<Labels>* labels;
	int D; 
	int N;
	int K;
	Float* Q_diag;
	vector<pair<int, Float>>* act_k_index;
	HashClass* hashfunc;
	int* hashindices;
	vector<int>* nnz_index;
	#ifdef USING_HASHVEC
	pair<int, Float>** w;
	int* size_w;
	pair<int, Float>** v;
	pair<int, Float>** alpha;
	int* size_v;
	int* util_v;
	#else
	Float** w;
	Float** v;
	#endif
		
	int max_iter;
	Float* grad;
	Float* Dk;
};
