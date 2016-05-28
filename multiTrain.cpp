#include "util.h"
#include "multi.h"
#include "SplitOracleActBCD.h"
#include "SBCDsolve.h"
//#include "ActBCDsolve.h"
//#include "OracleActBCD.h"
#include "PostSolve.h"
//#include <unistd.h>

double overall_time = 0.0;

void exit_with_help(){
	cerr << "Usage: ./multiTrain (options) [train_data] (model)" << endl;
	cerr << "options:" << endl;
	cerr << "-s solver: (default 0)" << endl;
	cerr << "	0 -- Stochastic Block Coordinate Descent" << endl;
	//cerr << "	1 -- Active Block Coordinate Descent" << endl;
	cerr << "	1 -- Stochastic-Active Block Coordinate Descent(PD-Sparse)" << endl;
	cerr << "-l lambda: L1 regularization weight (default 1.0)" << endl;
	cerr << "-c cost: cost of each sample (default 1.0)" << endl;
	cerr << "-r speed_up_rate: sample 1/r fraction of non-zero features to estimate gradient (default r = ceil(min( 5DK/(Clog(K)nnz(X)), nnz(X)/(5N) )) )" << endl;
	cerr << "-q split_up_rate: divide all classes into q disjoint subsets (default 1)" << endl;
	cerr << "-m max_iter: maximum number of iterations allowed (default 20)" << endl;
	cerr << "-u uniform_sampling: use uniform sampling instead of importance sampling (default not)" << endl;
	cerr << "-g max_select: maximum number of dual variables selected during search (default: -1 (i.e. dynamically adjusted during iterations) )" << endl;
	cerr << "-p post_train_iter: #iter of post-training without L1R (default 0)" << endl;
	cerr << "-e early_terminate (default 3)" << endl;
	cerr << "-h <file>: using heldout file <file>" << endl;
	exit(0);
}

/*size_t getTotalSystemMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (pages * page_size / 1024);
}*/

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
			case 'r': param->speed_up_rate = atoi(argv[i]);
				  break;
			case 'q': param->split_up_rate = atoi(argv[i]);
				  break;
			case 'm': param->max_iter = atoi(argv[i]);
				  break;
			case 'u': param->using_importance_sampling = false; --i;
				  break;
			case 'g': param->max_select = atoi(argv[i]);
				  break;
			case 'p': param->post_solve_iter = atoi(argv[i]);
				  break;
			case 'e': param->early_terminate = atoi(argv[i]);
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
	int D = train->D;
	int K = train->K;
	int N = train->data.size();
	cerr << "N=" << N << endl;
	cerr << "d=" << (Float)nnz(train->data)/N << endl;
	cerr << "D=" << D << endl; 
	cerr << "K=" << K << endl;
	
	/*
	#ifndef USING_HASHVEC
	//cerr << "Assume we are using Unix system! " << endl;
	size_t total_memory = getTotalSystemMemory();
	size_t max_need = D*K;
	if (max_need < N*K)
		max_need = N*K;
	//cout << total_memory/2 << " " << 2*sizeof(Float)*max_need/1024 << endl;
	if (total_memory/2 < (2*sizeof(Float)*max_need/1024)) {
		cerr << "WARNING: You might not have enough memory! try using multiTrainHash! " << endl;
		exit(0);
	}
	#endif
	*/
		
	if( param->solver == 0 ){
		
		SBCDsolve* solver = new SBCDsolve(param);
		Model* model = solver->solve();
		model->writeModel(param->modelFname);
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
	}else if( param->solver==1 ){
		SplitOracleActBCD* solver = new SplitOracleActBCD(param);
		Model* model = solver->solve();
		model->writeModel(param->modelFname);
		
		if( param->post_solve_iter > 0 ){
			
			param->post_solve_iter = min(solver->iter, param->post_solve_iter);
			#ifdef USING_HASHVEC
			PostSolve* postSolve = new PostSolve( param, model->w_hash_nnz_index, model->w, model->size_w, solver->act_k_index, solver->hashindices );
			#else
			PostSolve* postSolve = new PostSolve( param, model->w_hash_nnz_index, model->w, solver->act_k_index );
			#endif
			model = postSolve->solve();
	
			char* postsolved_modelFname = new char[FNAME_LEN];
			sprintf(postsolved_modelFname, "%s.p", param->modelFname);
			model->writeModel(postsolved_modelFname);
			delete[] postsolved_modelFname;
		}
	}
	
	overall_time += omp_get_wtime();
	cerr << "overall_time=" << overall_time << endl;
}
