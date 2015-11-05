all: train predict
	
train:
	g++ -fopenmp -std=c++11 -O3 -o multiTrain multiTrain.cpp
predict:
	g++ -fopenmp -std=c++11 -O3 -o multiPred multiPred.cpp
data20:
	./multiTrain -s 2 -m 70 data/data20.subtrain.svm 
	./multiPred data/data20.subtrain.svm model
	./multiPred data/data20.test.svm model
rcv1:
	./multiTrain -s 2 data/rcv1_train.multiclass 
	./multiPred data/rcv1_train.multiclass model
	./multiPred data/rcv1_test.multiclass.10k model
	
