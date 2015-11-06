#include "util.h"
#include "multi.h"
#include <cassert>

#define speed_up_rate 20

class OracleActBCD{
	
	public:
	OracleActBCD(Param* param){
		
		prob = param->prob;
		lambda = param->lambda;
		C = param->C;
		
		data = &(prob->data);
		labels = &(prob->labels);
		D = prob->D;
		N = prob->N;
		K = prob->K;

		max_iter = param->max_iter;
		max_select = param->max_select;
		tc1 = 0.0; tc2 = 0.0; tc3 = 0.0;
		prod = new double[K];
		for(int k=0;k<K;k++){
			prod[k] = 0.0;
			k_index.push_back(k);
		}
	}
	
	~OracleActBCD(){
		delete[] prod;
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
			for(int k=0;k<K;k++)
				v[j][k] = 0.0;
		}
		w = new HashVec*[D];
		for(int j=0;j<D;j++)
			w[j] = new HashVec();
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
		double* alpha_i_new = new double[K];
		int iter = 0;
		while( iter < max_iter ){
			
			random_shuffle( index, index+N );
			for(int r=0;r<N;r++){	
				
				int i = index[r];
				double* alpha_i = alpha[i];
				//search active variable
				search_time -= omp_get_wtime();
				//search_active_i( i, act_k_index[i]);
				search_active_i_approx( i, act_k_index[i], speed_up_rate );
				search_time += omp_get_wtime();
				//solve subproblem
				if( act_k_index[i].size() < 2 )
					continue;
				
				subsolve_time -= omp_get_wtime();
				subSolve(i, act_k_index[i], alpha_i_new);
				subsolve_time += omp_get_wtime();
				
				//maintain v =  X^T\alpha;  w = prox_{l1}(v);
				maintain_time -= omp_get_wtime();
				SparseVec* x_i = data->at(i);
				int yi = labels->at(i);
				for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
					
					int j = it->first;
					double f_val = it->second;
					HashVec* wj = w[j];
					for(vector<int>::iterator it=act_k_index[i].begin(); it!=act_k_index[i].end(); it++){
						int k = *it;
						double delta_alpha = (alpha_i_new[k]-alpha_i[k]);
						if( fabs(delta_alpha) < 1e-12 )
							continue;
						//update v
						double vjk_old = v[j][k]; 
						double vjk = vjk_old + f_val*delta_alpha;
						v[j][k] = vjk ;
						//update w
						double wjk_old = prox_l1(vjk_old,lambda);
						double wjk = prox_l1(vjk,lambda);
						if(  wjk_old ==0.0 && wjk==0.0 ){
							continue;
						}else if( wjk_old == 0.0 ){
							wj->insert(make_pair(k, (vjk>0.0)?(vjk-lambda):(vjk+lambda) ));
						}else if( wjk == 0.0 ){
							wj->erase(k);
						}else{
							wj->find(k)->second = wjk;
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
				double act_size_avg = 0.0;
				for(int i=0;i<N;i++){
					act_size_avg += act_k_index[i].size();	
				}
				act_size_avg /= N;
				cerr << "act="<< act_size_avg << endl;
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
			for(HashVec::iterator it=w[j]->begin(); it!=w[j]->end(); it++){
				d_obj += it->second*it->second;
			}
			nnz_w+=w[j]->size();
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
		for(int j=0;j<D;j++)
			delete[] v[j];
		delete[] v;
		cerr << "calc time=" << tc1 << " sort time=" << tc2 << " look up time=" << tc3 << endl;
		
		return new Model(prob, w); //v is w
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
			HashVec* wj = w[j];
			for( HashVec::iterator it2=wj->begin(); it2!=wj->end(); it2++ ){
				prod[it2->first] += it2->second * xij;
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
				if( num_select >= max_select )
					break;
			}
		}
		
		delete[] prod;
	}
		
	void search_active_i_approx( int i, vector<int>& act_k_index, int rate ){

                //compute <xi,wk> for k=1...K
                int yi = labels->at(i);
		memset(prod, 0, sizeof(double)*K);
                SparseVec* xi = data->at(i);
		int nnz = xi->size();
		for(int j = 0; j < act_k_index.size(); j++){
			prod[act_k_index[j]] = -10000000.0;
		}
		prod[yi] = -10000000.0;
		
		int n = 0;
		int counter = 0;
		tc1 -= omp_get_wtime();
		int max_index = 0;
		random_shuffle(xi->begin(), xi->end());
		while (n < nnz/rate) {
                	pair<int, double>* it = &(xi->at(n++));
		//for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
                //        int j = it->first;
                        double xij = it->second;
                        HashVec* wj = w[it->first];
			counter += wj->size();
                        for( HashVec::iterator it2=wj->begin(); it2!=wj->end(); it2++ ){
                                prod[it2->first] += it2->second * xij;
				if (prod[it2->first] > prod[max_index]){
					max_index = it2->first;
				}
			}
                }
		//found best over all updated prod
		tc1 += omp_get_wtime();
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
		tc1 += omp_get_wtime();
		//cerr << counter << "/" << K << endl;

                //sort accordingg to <xi,wk> in decreasing order
			
		/*tc2 -= omp_get_wtime();
		int k, bestk = 0;
		for (int k = 1; k < K; k++){
			if (prod[k] > prod[bestk]){
				bestk = k;
			}
		}
		if (prod[bestk] > th){
			act_k_index.push_back(bestk);
		}
		tc2 += omp_get_wtime();*/
		
		/*tc2 -= omp_get_wtime();
                sort(k_index.begin(), k_index.end(), ScoreComp(prod));
		
		int num_select=0;
		double th = -n/(1.0*nnz);
                for(int r=0;r<1;r++){
                        int k = k_index[r];
//                        if( alpha[i][k]<0.0 || k==yi ){ //exclude already in active set
//                                assert(prod[k] <= th);
//				continue;
//			}
                        if( prod[k] > th){
                                act_k_index.push_back(k);
                                num_select++;
                                if( num_select >= max_select )
                                        break;
                        } else {
				break;
			}
                }
		tc2 += omp_get_wtime();*/
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
	vector<int>* labels ;
	int D; 
	int N;
	int K;
	double* Q_diag;
	double* prod;
	double** alpha;
	double** v;
	HashVec** w;
	int max_iter;
	int max_select;
	
	vector<int> k_index;
};
