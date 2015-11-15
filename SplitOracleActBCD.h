#include "util.h"
#include "multi.h"
#include <cassert>
#define INFI 1e9
#define loc(k) k*split_up_rate/K

class SplitOracleActBCD{
	
	public:
	SplitOracleActBCD(Param* param){
		prob = param->prob;
		lambda = param->lambda;
		C = param->C;
		N = prob->N;
		D = prob->D;
		K = prob->K;
	
		//sampling 	
		speed_up_rate = param->speed_up_rate;	
		split_up_rate = param->split_up_rate;
		using_importance_sampling = param->using_importance_sampling;	
		max_select = param->max_select;
		prod = new double[K];
		
		data = &(prob->data);
		//compute l_1 norm of every feature x_i
		cdf_sum = new vector<double>();
		for(int i = 0; i < N; i++){
			SparseVec* xi = data->at(i);
			double _cdf = 0.0;
			for (SparseVec::iterator it = xi->begin(); it < xi->end(); it++){
				_cdf += fabs(it->second);
			}
			cdf_sum->push_back(_cdf);
		}
		labels = &(prob->labels);
		max_iter = param->max_iter;
		
		//DEBUG
		cccc= 0;
	}
	
	~SplitOracleActBCD(){
	}

	Model* solve(){
		//initialize alpha and v ( s.t. v = X^Talpha )
		alpha = new double*[N];
		for(int i=0;i<N;i++){
			alpha[i] = new double[K];
			for(int k=0;k<K;k++)
				alpha[i][k] = 0.0;
		}
		v = new double*[D]; //w = prox(v);
		for(int j=0;j<D;j++){
			v[j] = new double[K];
			for(int k=0;k<K;k++){
				v[j][k] = 0.0;
			}
		}
		w_nnz_index = new vector<int>**[D];
		for(int j=0;j<D;j++){
			w_nnz_index[j] = new vector<int>*[split_up_rate];
			for(int S=0;S < split_up_rate; S++){
				w_nnz_index[j][S] = new vector<int>();
			}
		}
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
		vector<int>* act_k_index = new vector<int>[N];
		for(int i=0;i<N;i++){
			Labels* yi = &(labels->at(i));
			for (Labels::iterator it = yi->begin(); it < yi->end(); it++){
				act_k_index[i].push_back(*it);
			}
		}
		
		//main loop
		double starttime = omp_get_wtime();
		double search_time=0.0, subsolve_time=0.0, maintain_time=0.0;
		double last_search_time = 0.0, last_subsolve_time = 0.0, last_maintain_time = 0.0;
		double* alpha_i_new = new double[K];
		int iter = 0;
		while( iter < max_iter ){
			
			random_shuffle( index, index+N );
			for(int r=0;r<N;r++){	
				
				int i = index[r];
				double* alpha_i = alpha[i];
				//search active variable
				search_time -= omp_get_wtime();
				if (using_importance_sampling)
					search_active_i_importance( i, act_k_index[i]);
				else
					search_active_i_uniform(i, act_k_index[i]);
				search_time += omp_get_wtime();
				//solve subproblem
				if( act_k_index[i].size() < 2 )
					continue;
					
				subsolve_time -= omp_get_wtime();
				subSolve(i, act_k_index[i], alpha_i_new);
				subsolve_time += omp_get_wtime();
				
				//maintain v =  X^T\alpha;  w = prox_{l1}(v);
				SparseVec* x_i = data->at(i);
				Labels* yi = &(labels->at(i));
				maintain_time -= omp_get_wtime();
				for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){	
					int j = it->first; 
					double f_val = it->second;
					for(vector<int>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
						int k = *it;
						double delta_alpha = (alpha_i_new[k]-alpha_i[k]);
						if( fabs(delta_alpha) < 1e-12 )
							continue;
						//update v
						int S = loc(k); // location of k in [split_up_rate]
						double vjk_old = v[j][k];
						double vjk = vjk_old + f_val*delta_alpha;
						v[j][k] = vjk;
						double wjk = prox_l1(vjk, lambda);
						double wjk_old = prox_l1(vjk_old, lambda);
						if ( wjk_old != wjk ){
                                                        if(  wjk_old == 0.0  ){
                                                                w_nnz_index[j][S]->push_back(k);
                                                        }
                                                }
					}
				}
				//update alpha
				bool has_zero=0;
				for(vector<int>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
					int k = *it;
					alpha_i[k] = alpha_i_new[k];
					has_zero |= (fabs(alpha_i[k])<1e-12);
				}
				//shrink act_k_index
				if( has_zero ){
					//cerr << "before size=" << act_k_index[i].size() << endl;
					vector<int> tmp_vec;
					tmp_vec.reserve(act_k_index[i].size());
					for(vector<int>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
						int k = *it;
					//	cerr << alpha_i[k] << " ";
						if( fabs(alpha_i[k]) > 1e-12 || find(yi->begin(), yi->end(), k)!=yi->end() )
							tmp_vec.push_back(k);
					}
					//cerr << endl;
					act_k_index[i] = tmp_vec;
					//cerr << "after size=" << act_k_index[i].size() << endl;
				}
				maintain_time += omp_get_wtime();
			}
			
			if( iter % 1 == 0 ){
				cerr << "." ;
				int nnz_a_i = 0;
				for(int i=0;i<N;i++){
					nnz_a_i += act_k_index[i].size();	
				}
				cerr << "nnz_a_i="<< ((double)nnz_a_i/N) << "  \t";
				int nnz_w_j = 0;
				for(int j=0;j<D;j++){
					for(int S=0;S < split_up_rate; S++){
						nnz_w_j += w_nnz_index[j][S]->size();
					}
				}
				cerr << "nnz_w_j=" << ((double)nnz_w_j/D) << "  \t";
				cerr << "search=" << search_time-last_search_time << "  \t";
				cerr << "subsolve=" << subsolve_time-last_subsolve_time << "  \t";
				cerr << "maintain=" << maintain_time-last_maintain_time << endl;
				last_search_time = search_time; last_subsolve_time = subsolve_time; last_maintain_time = maintain_time;
			}
				
			iter++;
		}
		double endtime = omp_get_wtime();
		cerr << endl;
		
		double d_obj = 0.0;
		int nSV = 0;
		int nnz_w = 0;
		int jk=0;
		for(int j=0;j<D;j++){
			for(int S=0;S<split_up_rate;S++){
				for (vector<int>::iterator it=w_nnz_index[j][S]->begin(); it!=w_nnz_index[j][S]->end(); it++){
					double wjk = prox_l1(v[j][*it], lambda);
					d_obj += wjk*wjk;//W[j][*it]*W[j][*it];
				}
				nnz_w+=w_nnz_index[j][S]->size();
			}
		}
		d_obj/=2.0;
		for(int i=0;i<N;i++){
			Labels* yi = &(labels->at(i));
			for(int k=0;k<K;k++){
				if(find(yi->begin(), yi->end(), k) == yi->end())
					d_obj += alpha[i][k];
				if( fabs( alpha[i][k] ) > 1e-12 )
					nSV++;
			}
		}
		cerr << "dual_obj=" << d_obj << endl;
		cerr << "nSV=" << nSV << " (NK=" << N*K << ")"<< endl;
		cerr << "nnz_w=" << nnz_w << " (DK=" << D*K << ")" << endl;
		cerr << "train time=" << endtime-starttime << endl;
		cerr << "search time=" << search_time << endl;
		cerr << "subsolve time=" << subsolve_time << endl;
		cerr << "maintain time=" << maintain_time << endl;
		//delete algorithm-specific variables
		delete[] alpha_i_new;
		delete[] act_k_index;
		for(int i=0;i<N;i++)
			delete[] alpha[i];
		delete[] alpha;
		delete[] Q_diag;
		delete[] prod;
		delete cdf_sum;
		w_temp = new HashVec*[D];
		for(int j = 0; j < D; j++){
			w_temp[j] = new HashVec();
			for(int S=0;S<split_up_rate;S++){
				for (vector<int>::iterator it=w_nnz_index[j][S]->begin(); it!=w_nnz_index[j][S]->end(); it++){
					int k = *it;
					w_temp[j]->insert(make_pair(k, prox_l1(v[j][k], lambda)));
				}
			}
		}
		for(int j=0;j<D;j++)
			delete[] v[j];
		delete[] v;
		return new Model(prob, w_temp); //v is w
	}
	int cccc;	
	void subSolve(int I, vector<int>& act_k_index, double* alpha_i_new){
		
		Labels* yi = &(labels->at(I));
		int m = yi->size(), n = act_k_index.size() - m;
		double* b = new double[n];
		double* c = new double[m];
		int* act_index_b = new int[n];
		int* act_index_c = new int[m];
		
		SparseVec* x_i = data->at(I);
		double A = Q_diag[I];
		double* alpha_i = alpha[I];
		//compute gradient of each k
		int i = 0, j = 0;
		//cerr << "act_size= " << act_k_index.size() << endl;
		//cerr << "(m,n)= " << m << "," << n << endl;
//		cerr << "alpha: ";
		for(int k=0;k<m+n;k++){
			int p = act_k_index[k];
	//		cerr << p << " ";
			if( find(yi->begin(), yi->end(), p) == yi->end() )
				act_index_b[i++] = p;
			else
				act_index_c[j++] = p;
//			cerr << alpha_i[p] << " ";
		}
//		cerr << endl;
		//cerr << endl;
		//cerr << "actual bar_Y_i size = " << i << endl;
		//cerr << "actual Y_i size = " << j << endl;
		assert(i==n); assert(j==m);
		int* index_b = new int[n];
		int* index_c = new int[m];
	//	cerr << "n= " << n << ",m= " << m << endl;
		for(int i=0; i < n; i++){
			int k = act_index_b[i];
			b[i] = 1.0 - A*alpha_i[k];
			index_b[i] = i;
		}

		for(int j=0; j < m; j++){ 
                        int k = act_index_c[j];
                        c[j] = A*alpha_i[k];
//			cerr << "update c to " << c[j] << endl;
			index_c[j] = j;
                }

		for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
			int fea_ind = it->first;
			double fea_val = it->second;
			for(int i = 0; i < n; i++){
				int k = act_index_b[i];
				double vjk = v[fea_ind][k];
				if (fabs(vjk) > lambda){
					if (vjk > 0)
						b[i] += (vjk-lambda)*fea_val;
					else
						b[i] += (vjk+lambda)*fea_val;
				}	
			}
			for(int j = 0; j < m; j++){
				int k = act_index_c[j];
				//grad[j] += W[fea_ind][k]*fea_val;
				double vjk = v[fea_ind][k];
				if( fabs(vjk) > lambda ){
					if( vjk > 0 )
						c[j] -= (vjk-lambda)*fea_val;
					else
						c[j] -= (vjk+lambda)*fea_val;
//					cerr << "update c to " << c[j] << " using (j,k)=" << fea_ind << "," << k  << " val=" << vjk << endl;
				}
			}
		}
			
		sort(index_b, index_b+n, ScoreComp(b));
		sort(index_c, index_c+m, ScoreComp(c));				
		double* S_b = new double[n];
		double* S_c = new double[m];
		double r_b = 0.0, r_c = 0.0;
//		cerr << "b: ";
		for (int i = 0; i < n; i++){
			b[index_b[i]] /= A;
//			cerr << b[index_b[i]] << " ";
			r_b += b[index_b[i]]*b[index_b[i]];
			if (i == 0)
				S_b[i] = b[index_b[i]];
			else
				S_b[i] = S_b[i-1] + b[index_b[i]];
		}
//		cerr << endl;
//		cerr << "c: ";
		for (int j = 0; j < m; j++){
                        c[index_c[j]] /= A;
//			cerr << c[index_c[j]] << " ";
			r_c += c[index_c[j]]*c[index_c[j]];
			if (j == 0)
				S_c[j] = c[index_c[j]];
			else
                        	S_c[j] = S_c[j-1] + c[index_c[j]];
                }
//		cerr << endl;
		i = 0; j = 0; 
		while (i < n && S_b[i] - (i+1)*b[index_b[i]] <= 0){
			r_b -= b[index_b[i]]*b[index_b[i]];
			i++;
		} 
		while (j < m && S_c[j] - (j+1)*c[index_c[j]] <= 0) { 
                        r_c -= c[index_c[j]]*c[index_c[j]];
                        j++;
                }
		//update for b_{0..i-1} c_{0..j-1}
		//i,j is the indices of coordinate that we will going to include, but not now!
		//cerr << "init i,j = " << i << "," << j << endl;
		double t = 0.0;
		double ans_t_star = 0; //(sqrt(i)*S_c[j-1] + sqrt(j)*S_b[i-1])/(sqrt(i)+sqrt(j));
		double ans = INFI; //r_b + r_c + (S_b[i-1] - ans_t_star)*(S_b[i-1] - ans_t_star)/i + (S_c[j-1] - ans_t_star)*(S_c[j-1] - ans_t_star)/j;
		int ansi = i, ansj = j;
		int lasti = 0, lastj = 0;
		do{
			lasti = i; lastj = j;
			// l = t; t = min(f_b(i), f_c(j));
			double l = t;
			if (i == n){
				if (j == m){
					t = C;
				} else {
					t = S_c[j] - (j+1)*c[index_c[j]];
				}
			} else {
				if (j == m){
					t = S_b[i] - (i+1)*b[index_b[i]];
				} else {
					t = S_b[i] - (i+1)*b[index_b[i]];
					if (S_c[j] - (j+1)*c[index_c[j]] < t){ 
                                		t = S_c[j] - (j+1)*c[index_c[j]]; 
                        		}
				}
			}
			if (t > C){
				t = C;
			}
//			cerr << l << " " << t << endl;
			assert(t >= 0 && l >= 0);
		//	cerr << "set t = " << t << endl;
			double t_star = (i*S_c[j-1] + j*S_b[i-1])/(i+j);
		//	cerr << "get t_star = " << t_star << endl;
			if (t_star < l){
				t_star = l;
			}
			if (t_star > t){
				t_star = t;
			}
			double candidate = r_b + r_c + (S_b[i-1] - t_star)*(S_b[i-1] - t_star)/i + (S_c[j-1] - t_star)*(S_c[j-1] - t_star)/j;
			if (candidate < ans){
				ans = candidate;
				ansi = i;
				ansj = j;
				ans_t_star = t_star;
			}
			while (i < n && S_b[i] - (i+1)*b[index_b[i]] <= t){
               	         	r_b -= b[index_b[i]]*b[index_b[i]];
               	        	i++;
               	 	}
               	 	while (j < m && S_c[j] - (j+1)*c[index_c[j]] <= t) {
               	         	r_c -= c[index_c[j]]*c[index_c[j]];
               	        	j++;
               	 	}
		} while (i != lasti || j != lastj);
		i = ansi; j = ansj;
		for(int i = 0; i < n; i++){
			int k = act_index_b[index_b[i]];
			if (i < ansi)
				alpha_i_new[k] = -(b[index_b[i]] + (ans_t_star - S_b[ansi-1])/ansi);
			else
				alpha_i_new[k] = 0.0;
//			cerr << I << " " << k  << " " << alpha_i_new[k] << " ";
		}
//		cerr << endl;
		for(int j = 0; j < m; j++){
                        int k = act_index_c[index_c[j]];
			if (j < ansj)
                        	alpha_i_new[k] = c[index_c[j]] + (ans_t_star - S_c[ansj-1])/ansj;
			else
				alpha_i_new[k] = 0.0;
//			cerr << I << " " << k << " " << alpha_i_new[k] << " ";
                }
//		cerr << endl;
//		cerr << "ans="<< ans << " i_star="<< ansi << " j_star=" << ansj << " t_star=" << ans_t_star << endl;	
//		cerr << endl;
//		cccc++;
//		cerr << cccc << endl;
//		if (cccc == 15580)
//			exit(0);
		/*for(int j=0;j<act_k_size;j++){
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
		}*/
	
			
		/*delete[] b; delete[] c;
		delete[] act_index_b; delete[] act_index_c;
		delete[] S_b; delete[] S_c;
		delete[] alpha_i;
		delete[] index_b; delete[] index_c; */
		//delete[] Dk;
	}
		
	void search_active_i_importance( int i, vector<int>& act_k_index ){
		int S = rand()%split_up_rate;
                //compute <xi,wk> for k=1...K
                Labels* yi = &(labels->at(i));
		memset(prod, 0, sizeof(double)*K);
                SparseVec* xi = data->at(i);
		int nnz = xi->size();
		for(int j = 0; j < act_k_index.size(); j++){
			prod[act_k_index[j]] = -INFI;
		}
		for (Labels::iterator it = yi->begin(); it < yi->end(); it++){
			prod[*it] = -INFI;
		}
		int n = nnz/speed_up_rate;
		double th = -n/(1.0*nnz);
		vector<double>* rand_nums = new vector<double>();
		for (int tt = 0; tt < n; tt++){
			rand_nums->push_back(((double)rand()/(RAND_MAX)));
		}
		sort(rand_nums->begin(), rand_nums->end()); 
		#ifdef MULTISELECT
		int* max_indices = new int[max_select+1];
		for(int ind = 0; ind <= max_select; ind++){
			max_indices[ind] = -1;
		}
		#endif
		#ifndef MULTISELECT
		int max_index = 0;
		#endif
		SparseVec::iterator current_index = xi->begin();
		double current_sum = current_index->second;
		vector<double>::iterator current_rand_index = rand_nums->begin();
		double cdf_sumi = cdf_sum->at(i);
			
		while (current_rand_index < rand_nums->end()){
			while (current_sum < (*current_rand_index)*cdf_sumi){
				current_index++;
				current_sum += current_index->second;
			}
			double xij = 0.0;
			while (current_rand_index < rand_nums->end() && current_sum >= (*current_rand_index)*cdf_sumi ){
				xij = xij + 1.0;
				current_rand_index++;
			}
                        xij *= cdf_sumi*((current_index->second > 0.0)?1:(-1));
			int j = current_index->first;
			vector<int>* wj = w_nnz_index[j][S];
			int k = 0, ind = 0;
			double wjk = 0.0;
                        for(vector<int>::iterator it2 = wj->begin(); it2<wj->end(); it2++ ){
				k = *it2;
				wjk = prox_l1(v[j][k], lambda);
				if (wjk == 0.0){
					*it2=*(wj->end()-1);
					wj->erase(wj->end()-1);
					it2--;
					continue;
				}
                                prod[k] += wjk * xij;
				#ifndef MULTISELECT
				if (prod[max_index] < prod[k]){
					max_index = k;
				}
				#endif
				#ifdef MULTISELECT
				if (prod[k] > th){
					ind = 0;
					while (ind < max_select && max_indices[ind] != -1 && max_indices[ind] != k){
						ind++;
					}
					max_indices[ind] = k;
					//try move to left
					while (ind > 0 && prod[max_indices[ind]] > prod[max_indices[ind-1]]){
						//using k as temporary variables
						k = max_indices[ind];
						max_indices[ind] = max_indices[ind-1];
						max_indices[ind-1] = k;
					}
					//try move to right
					while (ind < max_select-1 && max_indices[ind+1] != -1 && prod[max_indices[ind+1]] > prod[max_indices[ind]]){
                                                //using k as temporary variables
                                                k = max_indices[ind];
                                                max_indices[ind] = max_indices[ind+1];
                                                max_indices[ind+1] = k;
                                        }
				}
				#endif
			}
                }
		rand_nums->clear();
		#ifdef MULTISELECT
		for (int j = 0; j < max_select; j++){
			if (max_indices[j] != -1 && prod[max_indices[j]] > 0.0) 
				continue;
			for (int r = 0; r < K; r++){
				int k = rand() % K;
				if (prod[k] == 0){
					bool flag = false;
					for (int ind = 0; ind < max_select; ind++){
						if (max_indices[ind] == k){
							flag = true;
							break;
						}
					}
					if (!flag){
						max_indices[j] = k;
						break;
					}
				}
			}
		}
		for(int ind = 0; ind < max_select; ind++){
			if (max_indices[ind] != -1 && prod[max_indices[ind]] > th){
				act_k_index.push_back(max_indices[ind]);
			}
		}
		#endif
		#ifndef MULTISELECT
		if (prod[max_index] < 0){
			for (int r = 0; r < K; r++){
				int k = rand() % K;
				if (prod[k] == 0){
					max_index = k;
					break;
				}
			}
		}
		if (prod[max_index] > th){
			for (int k = 0; k < act_k_index.size(); k++){
				//cerr << prod[max_index] << " " << th<< endl;
				assert(act_k_index[k] != max_index);
			}
			act_k_index.push_back(max_index);
//			cerr << "yes" << endl;
		} else{
//			cerr << "no" << endl;
		}
 
		#endif
        }

	void search_active_i_uniform(int i, vector<int>& act_k_index){
		int S = rand()%split_up_rate;
		Labels* yi = &(labels->at(i));
                memset(prod, 0, sizeof(double)*K);
                SparseVec* xi = data->at(i);
                int nnz = xi->size();
                for(int j = 0; j < act_k_index.size(); j++){
                        prod[act_k_index[j]] = -INFI;
                }
		for(Labels::iterator it = yi->begin(); it < yi->end(); it++){
                	prod[*it] = -INFI;
		}
                int n = nnz/speed_up_rate;
                vector<double>* rand_nums = new vector<double>();
                for (int tt = 0; tt < n; tt++){
                        rand_nums->push_back(((double)rand()/(RAND_MAX)));
                }
                sort(rand_nums->begin(), rand_nums->end());
                int max_index = 0;
		random_shuffle(xi->begin(), xi->end());
                SparseVec::iterator current_index = xi->begin();
                for (int t = 0; t < n; t++){
                        double xij = current_index->second;
                        int j = current_index->first;
                        current_index++;
                        vector<int>* wj = w_nnz_index[j][S];
                        int k = 0;
                        double wjk = 0.0;
                        for(vector<int>::iterator it2 = wj->begin(); it2<wj->end(); it2++ ){
                                k = *it2;
                                wjk = prox_l1(v[j][k], lambda);
                                if (wjk == 0.0){
                                        *it2=*(wj->end()-1);
                                        wj->erase(wj->end()-1);
                                        it2--;
                                        continue;
                                }
                                prod[k] += wjk * xij;
                                if (prod[k] > prod[max_index]){
                                        max_index = k;
                                }
			}
		}
		double th = -n/(1.0*nnz);
                if (prod[max_index] < 0){
                        for(int k = 0; k < K; k++){
                                int r = rand()%K;
                                if (prod[r] == 0){
                                        max_index = r;
                                        break;
                                }
                        }
                }
		if (prod[max_index] > th){
                        act_k_index.push_back(max_index);
                }
	}
	
	private:
	Problem* prob;
	double lambda;
	double C;
	vector<SparseVec*>* data;
	vector<Labels>* labels;
	int D; 
	int N;
	int K;
	double* Q_diag;
	double** alpha;
	double** v;
	vector<double>* cdf_sum;
	HashVec** w_temp;
	vector<int>*** w_nnz_index;
	int max_iter;
	vector<int>* k_index;
	
	//sampling 
	bool using_importance_sampling;
	int max_select;
	int speed_up_rate, split_up_rate;	
	double* prod;
};
