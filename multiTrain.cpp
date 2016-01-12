#include "util.h"
#include "multi.h"
#include "SplitOracleActBCD.h"
#include "SBCDsolve.h"
//#include "ActBCDsolve.h"
//#include "OracleActBCD.h"
#include "PostSolve.h"
#include <unistd.h>

double overall_time = 0.0;

void exit_with_help(){
	cerr << "Usage: ./multiTrain (options) [train_data] (model)" << endl;
	cerr << "options:" << endl;
	cerr << "-s solver: (default 0)" << endl;
	cerr << "	0 -- Stochastic Block Coordinate Descent" << endl;
	//cerr << "	1 -- Active Block Coordinate Descent" << endl;
	cerr << "	3 -- Stochastic-Active Block Coordinate Descent" << endl;
	cerr << "-l lambda: L1 regularization weight (default 1.0)" << endl;
	cerr << "-c cost: cost of each sample (default 1)" << endl;
	cerr << "-r speed_up_rate: using 1/r fraction of samples (default min(max(DK/(log(K)nnz(X)),1),d/5) )" << endl;
	cerr << "-q split_up_rate: choose 1/q fraction of [K]" << endl;
	cerr << "-m max_iter: maximum number of iterations allowed (default 20)" << endl;
	cerr << "-i im_sampling: Importance sampling instead of uniform (default not)" << endl;
	cerr << "-g max_select: maximum number of greedy-selected dual variables per sample (default 1)" << endl;
	cerr << "-p post_train_iter: #iter of post-training w/o L1R (default 0)" << endl;
	
	exit(0);
}

size_t getTotalSystemMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (pages * page_size / 1024);
}

void parse_cmd_line(int argc, char** argv, Param* param){

	int i;
	for(i=1;i<argc;i++){
		if( argv[i][0] != '-' )
			break;
		if( ++i >= argc )
			exit_with_help();

		switch(argv[i-1][1]){
			
			case 's': param->solver = atoi(argv[i]);
				  break;
			case 'l': param->lambda = atof(argv[i]);
				  break;
			case 'c': param->C = atof(argv[i]);
				  break;
			case 'm': param->max_iter = atoi(argv[i]);
				  break;
			case 'g': param->max_select = atoi(argv[i]);
				  break;
			case 'r': param->speed_up_rate = atoi(argv[i]);
				  break;
			case 'i': param->using_importance_sampling = true; --i;
				  break;
			case 'q': param->split_up_rate = atoi(argv[i]);
				  break;
			case 'p': param->post_solve_iter = atoi(argv[i]);
				  break;
			case 'h': param->heldoutFname = argv[i];
				  break;
			default:
				  cerr << "unknown option: -" << argv[i-1][1] << endl;
				  exit(0);
		}
	}

	if(i>=argc)
		exit_with_help();

	param->trainFname = argv[i];
	i++;

	if( i<argc )
		param->modelFname = argv[i];
	else{
		param->modelFname = new char[FNAME_LEN];
		strcpy(param->modelFname,"model");
	}
}

void writeModel( char* fname, Model* model){

	ofstream fout(fname);
	fout << "nr_class " << model->K << endl;
	fout << "label ";
	for(vector<string>::iterator it=model->label_name_list->begin();
			it!=model->label_name_list->end(); it++)
		fout << *it << " ";
	fout << endl;
	fout << "nr_feature " << model->D << endl;
	int D = model->D;
	int K = model->K;
	for(int j=0;j<D;j++){
		vector<int>* nnz_index_j = model->w_hash_nnz_index[j];
		fout << nnz_index_j->size() << " ";
		#ifdef USING_HASHVEC
		int count = 0;
		pair<int, float_type>* wj = model->w[j];
		int size_wj = model->size_w[j];
		for (int it = 0; it < size_wj; it++){
			int k = wj[it].first;
			if (k == -1 || wj[it].second == 0.0)
				continue;
			count++;
			fout << k << ":" << wj[it].second << " ";
		//	cout << j << ":" << k << ":" << wj[it].second << endl;
		}
		assert(count == nnz_index_j->size());
		#else
		float_type* wj = model->w[j];
		for(vector<int>::iterator it=nnz_index_j->begin(); it!=nnz_index_j->end(); it++){
			fout << *it << ":" << wj[*it] << " ";
		}
		#endif
		fout << endl;
	}
	fout.close();
}

int main(int argc, char** argv){
	
	Param* param = new Param();
	parse_cmd_line(argc, argv, param);
	Problem* train = new Problem();
	readData( param->trainFname, train);
	param->train = train;
	
	overall_time -= omp_get_wtime();

	//param->lambda /= prob->N;
	if (param->heldoutFname != NULL){
		//assert(param->train->label_index_map == param->heldout->label_index_map);
		//assert(param->train->label_name_list == param->heldout->label_name_list);
		Problem* heldout = new Problem();
		readData( param->heldoutFname, heldout);
		cerr << "heldout N=" << heldout->data.size() << endl;
		//cerr << "heldout D=" << heldout->D << endl;
		//cerr << "heldout K=" << heldout->K << endl;
		param->heldoutEval = new HeldoutEval(heldout);
	}
	long nnz_X = nnz(train->data);
	int D = train->D;
	int K = train->K;
	int N = train->data.size();
	int d = nnz_X/N;
	cerr << "N=" << N << endl;
	cerr << "d=" << d << endl;
	cerr << "D=" << D << endl; 
	cerr << "K=" << K << endl;
	#ifndef USING_HASHVEC
	//cerr << "Assume we are using Unix system! " << endl;
	size_t total_memory = getTotalSystemMemory();
	size_t max_need = D*K;
	if (max_need < N*K)
		max_need = N*K;
	//cout << total_memory/2 << " " << 2*sizeof(float_type)*max_need/1024 << endl;
	if (total_memory/2 < (2*sizeof(float_type)*max_need/1024)) {
		cerr << " not enough total memory! try make with -DUSING_HASHVEC! " << endl;
		exit(0);
	}
	#endif
	if( param->speed_up_rate==-1 )
		param->speed_up_rate = (int)min( max(5.0*D*K/nnz_X/param->C/log((float_type)K), 1.0), d/20.0);
	cerr << "lambda=" << param->lambda << ", C=" << param->C << ", r=" << param->speed_up_rate  << endl;
	
	
		
	if( param->solver == 0 ){
		
		SBCDsolve* solver = new SBCDsolve(param);
		Model* model = solver->solve();
		writeModel(param->modelFname, model);
//	}else if( param->solver==1 ){
//		
//		ActBCDsolve* solver = new ActBCDsolve(param);
//		Model* model = solver->solve();
//		writeModel(param->modelFname, model);
//	}else if( param->solver==2 ){
//		
//		OracleActBCD* solver = new OracleActBCD(param);
//		Model* model = solver->solve();
//		writeModel(param->modelFname, model);
	}else if( param->solver==3 ){
		SplitOracleActBCD* solver = new SplitOracleActBCD(param);
		Model* model = solver->solve();
		writeModel(param->modelFname, model);
		
		if( param->post_solve_iter > 0 ){
			
			param->post_solve_iter = min(solver->iter, param->post_solve_iter);
			#ifdef USING_HASHVEC
			PostSolve* postSolve = new PostSolve( param, model->w_hash_nnz_index, model->w, model->size_w, solver->act_k_index, solver->v, solver->size_v, solver->hashindices );
			#else
			PostSolve* postSolve = new PostSolve( param, model->w_hash_nnz_index, model->w, solver->act_k_index, solver->v );
			#endif
			model = postSolve->solve();
			
			char* postsolved_modelFname = new char[FNAME_LEN];
			sprintf(postsolved_modelFname, "%s.p", param->modelFname);
			writeModel(postsolved_modelFname, model);
			delete[] postsolved_modelFname;
		}
	}
	
	overall_time += omp_get_wtime();
	cerr << "overall_time=" << overall_time << endl;
}
