all: train predict
	
train:
	g++ -fopenmp -std=c++11 -O3 -o multiTrain multiTrain.cpp
predict:
	g++ -fopenmp -std=c++11 -O3 -o multiPred multiPred.cpp

s=2
r=1
m=20
model=model
LSHTC=/scratch/cluster/ianyen/data//LSHTC/LSHTC1/large_lshtc_dataset/Task1_Train\:CrawlData_Test\:CrawlData/
sample_opt=-i

ocr:
	./multiTrain -s $(s) -r $(r) -m $(m) $(sample_opt) data/data20.subtrain.svm $(model)
	./multiPred data/data20.subtrain.svm $(model)
	./multiPred data/data20.test.svm $(model)
rcv1:
	./multiTrain -s $(s) -r $(r) data/rcv1_train.multiclass $(model)
	./multiPred data/rcv1_train.multiclass $(model)
	./multiPred data/rcv1_test.multiclass.10k $(model)
aloi:
	./multiTrain -s 2 -l 0.01 -m 20 data/aloi.bin.subtrain $(model)
	./multiPred data/aloi.bin.subtrain $(model)
	./multiPred data/aloi.bin.test $(model)

LSHTC:
	./multiTrain -s $(s) $(sample_opt) -l 0.025 -c 1 -r $(r) -m $(m) ${LSHTC}/train.tfidf.scale $(model)
	./multiPred ${LSHTC}/train.tfidf.scale $(model)
	./multiPred ${LSHTC}/val.tfidf.scale  $(model)
