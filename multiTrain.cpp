#include "util.h"
#include "multi.h"
#include "SBCDsolve.h"
#include "ActBCDsolve.h"
#include "OracleActBCD.h"

void exit_with_help(){

	cerr << "Usage: ./multiTrain (options) [train_data] (model)" << endl;
	cerr << "options:" << endl;
	cerr << "-s solver: (default 0)" << endl;
	cerr << "	0 -- Stochastic Block Coordinate Descent" << endl;
	cerr << "	1 -- Active Block Coordinate Descent" << endl;
	cerr << "	2 -- Oracle-Active Block Coordinate Descent" << endl;
	cerr << "-l lambda: L1 regularization weight (default 1.0)" << endl;
	cerr << "-c cost: cost of each sample (default 10)" << endl;
	cerr << "-r speed_up_rate: using 1/r fraction of samples (default 1)" << endl;
	cerr << "-m max_iter: maximum number of iterations allowed (default 20)" << endl;
	cerr << "-g max_select: maximum number of greedy-selected dual variables per sample (default 1)" << endl;
	exit(0);
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
	HashVec** w = model->w;
	int D = model->D;
	int K = model->K;
	for(int j=0;j<D;j++){
		fout << w[j]->size() << " ";
		for(HashVec::iterator it=w[j]->begin(); it!=w[j]->end(); it++)
			fout << it->first << ":" << it->second << " ";
		fout << endl;
	}
	fout.close();
}

int main(int argc, char** argv){
	
	Param* param = new Param();
	parse_cmd_line(argc, argv, param);
	
	Problem* prob = new Problem();
	readData( param->trainFname, prob);
	param->prob = prob;

	cerr << "N=" << prob->data.size() << endl;
	cerr << "D=" << prob->D << endl; 
	cerr << "K=" << prob->K << endl;
	cerr << "lambda=" << param->lambda << ", C=" << param->C << endl;
	//param->lambda /= prob->N;
	
	if( param->solver == 0 ){
		
		SBCDsolve* solver = new SBCDsolve(param);
		Model* model = solver->solve();
		writeModel(param->modelFname, model);
	}else if( param->solver==1 ){
		
		ActBCDsolve* solver = new ActBCDsolve(param);
		Model* model = solver->solve();
		writeModel(param->modelFname, model);
	}else if( param->solver==2 ){
		
		OracleActBCD* solver = new OracleActBCD(param);
		Model* model = solver->solve();
		writeModel(param->modelFname, model);
	}
}
