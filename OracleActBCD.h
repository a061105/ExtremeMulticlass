#include "util.h"
#include "multi.h"
#include <cassert>

#define INFI 1e9

class OracleActBCD{
	
	public:
	OracleActBCD(Param* param){
		
		prob = param->prob;
		lambda = param->lambda;
		C = param->C;
		speed_up_rate = param->speed_up_rate;	
		using_importance_sampling = param->using_importance_sampling;	
		data = &(prob->data);
		N = prob->N;
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
		D = prob->D;
		K = prob->K;

		max_iter = param->max_iter;
		max_select = param->max_select;
		prod = new double[K];
		for(int k=0;k<K;k++){
			k_index.push_back(k);
		}
		tc1 = 0.0; tc2 = 0.0; tc3 = 0.0;
	}
	
	~OracleActBCD(){
		delete[] prod;
		delete[] cdf_sum;
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
		
		/*W = new double*[D];
		for(int j=0;j<D;j++){
			W[j] = new double[K];
			for(int k=0;k<K;k++){
				W[j][k] = 0.0;
			}
		}*/
		//v_hash = new HashVec*[D];
		//for(int j = 0; j < D; j++){
		//	v_hash[j] = new HashVec();
		//}
		w_nnz_index = new vector<int>*[D];
		for(int j=0;j<D;j++){
			w_nnz_index[j] = new vector<int>();
			w_nnz_index[j]->reserve(100);
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
		for(int i=0;i<N;i++)
			act_k_index[i].push_back(labels->at(i));
		
		
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
				int yi = labels->at(i);
				maintain_time -= omp_get_wtime();
				for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){	
					int j = it->first;
					double f_val = it->second;
					//int len = 0;
					for(vector<int>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
						int k = *it;
						double delta_alpha = (alpha_i_new[k]-alpha_i[k]);
						if( fabs(delta_alpha) < 1e-12 )
							continue;
						//update v
						double vjk_old = v[j][k];
						double vjk = vjk_old + f_val*delta_alpha;
						v[j][k] = vjk;
						/*HashVec::iterator it3 = (v_hash[j]->find(k));
						if (it3 != v_hash[j]->end()){
							it3->second = vjk;
						} else {
							v_hash[j]->insert(make_pair(k, vjk));
						}*/
						double wjk = prox_l1(vjk, lambda);
						//double wjk_old = W[j][k];
						double wjk_old = prox_l1(vjk_old, lambda);
						if ( wjk_old != wjk ){
                                                        //W[j][k] = wjk;
                                                        if(  wjk_old == 0.0  ){
                                                                w_nnz_index[j]->push_back(k);
                                                        }
                                                }
						//update_k[len] = k;
						//update_wjk[len++] = (vjk > lambda)? (vjk-lambda):((vjk < -lambda)?(vjk + lambda):0 );
					}
					/*for(vector<int>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
						//update w
						int k = *it;
						//int k = update_k[ll];
						//double wjk = update_wjk[ll];
						//double wjk_old = W[j][k];
					}*/
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
					vector<int> tmp_vec;
					tmp_vec.reserve(act_k_index[i].size());
					for(vector<int>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
						int k = *it;
						if( fabs(alpha_i[k]) > 1e-12 || k==yi )
							tmp_vec.push_back(k);
					}
					act_k_index[i] = tmp_vec;
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
					nnz_w_j += w_nnz_index[j]->size();
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
			for (vector<int>::iterator it=w_nnz_index[j]->begin(); it!=w_nnz_index[j]->end(); it++){
				double wjk = prox_l1(v[j][*it], lambda);
				d_obj += wjk*wjk;//W[j][*it]*W[j][*it];
			}
			nnz_w+=w_nnz_index[j]->size();
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
		//cerr << "calc time=" << tc1 << " sort time=" << tc2 << " look up time=" << tc3 << endl;
		w_temp = new HashVec*[D];
		for(int j = 0; j < D; j++){
			w_temp[j] = new HashVec();
			for (vector<int>::iterator it=w_nnz_index[j]->begin(); it!=w_nnz_index[j]->end(); it++){
				int k = *it;
				w_temp[j]->insert(make_pair(k, prox_l1(v[j][k], lambda)));
			}
		}
		for(int j=0;j<D;j++)
			delete[] v[j];
		delete[] v;
		return new Model(prob, w_temp); //v is w
	}
	
	void subSolve(int i, vector<int>& act_k_index, double* alpha_i_new){
		
		int act_k_size = act_k_index.size();
		
		double* grad = new double[act_k_size];
		double* Dk = new double[act_k_size];
		
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
			int fea_ind = it->first;
			double fea_val = it->second;
			for(int j=0;j<act_k_size;j++){
				int k = act_k_index[j];
				//grad[j] += W[fea_ind][k]*fea_val;
				double vjk = v[fea_ind][k];
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

	void search_active_i( int i, vector<int>& act_k_index ){
	
		//compute <xi,wk> for k=1...K
		double* prod = new double[K];
		for(int k=0;k<K;k++)
			prod[k] = 0.0;
		SparseVec* xi = data->at(i);
		int yi = labels->at(i);
		for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
			int j = it->first;
			double xij = it->second;
			vector<int>* wj = w_nnz_index[j];
			int k = 0;
			for( vector<int>::iterator it2=wj->begin(); it2!=wj->end(); it2++ ){
				k = *it2;
				prod[k] += prox_l1(v[j][k],lambda) * xij;
			}
		}
		
		//sort accordingg to <xi,wk>
		sort(k_index.begin(), k_index.end(), ScoreComp(prod));
		int num_select=0;
		for(int r=0;r<K;r++){
			int k = k_index[r];
			if( alpha[i][k]<0.0 || k==yi ) //exclude already in active set
				continue;
			if( prod[k] > -1.0 ){
				act_k_index.push_back(k);
				num_select++;
				if( num_select >= max_select)
					break;
			}
		}
		
		delete[] prod;
	}
		
	void search_active_i_importance( int i, vector<int>& act_k_index ){
                //compute <xi,wk> for k=1...K
                int yi = labels->at(i);
		memset(prod, 0, sizeof(double)*K);
                SparseVec* xi = data->at(i);
		int nnz = xi->size();
		for(int j = 0; j < act_k_index.size(); j++){
			prod[act_k_index[j]] = -INFI;
		}
		prod[yi] = -INFI;
		int n = nnz/speed_up_rate;
		vector<double>* rand_nums = new vector<double>();
		for (int tt = 0; tt < n; tt++){
			rand_nums->push_back(((double)rand()/(RAND_MAX)));
		}
		sort(rand_nums->begin(), rand_nums->end()); 
		int max_index = 0;
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
			vector<int>* wj = w_nnz_index[j];
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

	void search_active_i_uniform(int i, vector<int>& act_k_index){
		int yi = labels->at(i);
                memset(prod, 0, sizeof(double)*K);
                SparseVec* xi = data->at(i);
                int nnz = xi->size();
                for(int j = 0; j < act_k_index.size(); j++){
                        prod[act_k_index[j]] = -INFI;
                }
                prod[yi] = -INFI;
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
                        vector<int>* wj = w_nnz_index[j];
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
	
	
	
	/*void searchActive( double* v, vector<int>* act_k_index){
		
		// convert v to w (in SparseMtrix format)
		SparseVec* w = new SparseVec[D];
		for(int j=0;j<D;j++){
			w[j].reserve(K);
			int j_offset = j*K;
			for(int k=0;k<K;k++){
				double v_jk = v[j_offset + k];
				if( fabs(v_jk) > lambda ){
					if(v_jk>0.0) w[j].push_back(make_pair(k,v_jk-lambda));
					else	     w[j].push_back(make_pair(k,v_jk+lambda));
				}	
			}
		}
		
		// Row-wise Sparse Matrix Multiplication X*W (x_i*W, i=1...N)
		double* prod_xi_w = new double[K];
		int* k_index = new int[K];
		for(int k=0;k<K;k++)
			k_index[k] = k;
		for(int i=0;i<N;i++){
			SparseVec* xi = data->at(i);
			int yi = labels->at(i);

			for(int k=0;k<K;k++)
				prod_xi_w[k] = 0.0;
			for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
				int j = it->first;
				double xij = it->second;
				for(SparseVec::iterator it2=w[j].begin(); it2!=w[j].end(); it2++){
					prod_xi_w[it2->first] += xij*it2->second;
				}
			}	

			sort(k_index, k_index+K, ScoreComp(prod_xi_w));
			int num_select=0;
			for(int r=0;r<K;r++){
				int k = k_index[r];
				if( k == yi || alpha[i][k]<0.0 )
					continue;
				if( prod_xi_w[k] > -1.0 ){
					act_k_index[i].push_back(k);
					num_select++;
					if( num_select >= max_select )
						break;
				}
			}
		}
		
		delete[] k_index;
		delete[] prod_xi_w;
		delete[] w;
	}*/
	
	private:
	double tc1, tc2, tc3;
	Problem* prob;
	double lambda;
	double C;
	vector<SparseVec*>* data ;
	vector<int>* labels;
	int D; 
	int N;
	int K;
	double* Q_diag;
	double* prod;
	double** alpha;
	double** v;
	//vector<SparseVec*>* cdf_x;
	vector<double>* cdf_sum;
	HashVec** w_temp;
	vector<int>** w_nnz_index;
	int max_iter;
	int max_select;
	int speed_up_rate;	
	vector<int> k_index;
	bool using_importance_sampling;
};
