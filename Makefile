all: train predict
	
FLAG= -DUSING_HASHVEC -DMULTISELECT
train:
	g++ -fopenmp -std=c++11 -O3 -o multiTrain multiTrain.cpp $(FLAG)
predict:
	g++ -fopenmp -std=c++11 -O3 -o multiPred multiPred.cpp

s=3
r=1
m=20
q=1
g=1
p=20
l=1
model=model
sample_opt=-i
data_dir=/scratch/cluster/ianyen/data

ocr:
	./multiTrain -s $(s) -r $(r) -m $(m) -q $(q) -g $(g) -p $(p) $(sample_opt) $(data_dir)/data/data20.subtrain.svm $(model)
	./multiPred $(data_dir)/data/data20.subtrain.svm $(model)
	./multiPred $(data_dir)/data/data20.test.svm $(model)

rcv1:
	./multiTrain -l $(l) -s $(s) -r $(r) -m $(m) -q $(q) -g $(g) -p $(p) $(sample_opt) -h $(data_dir)/rcv1_test.multiclass.10k $(data_dir)/rcv1_train.multiclass $(model)
	./multiPred $(data_dir)/rcv1_train.multiclass $(model)
	./multiPred $(data_dir)/rcv1_test.multiclass.10k $(model)
ifneq ($(p), 0)
	./multiPred $(data_dir)/rcv1_train.multiclass $(model).p
	./multiPred $(data_dir)/rcv1_test.multiclass.10k $(model).p
endif

sector:
	./multiTrain -l $(l) -s $(s) -r $(r) -m $(m) -q $(q) -g $(g) -p $(p) $(sample_opt) -h $(data_dir)/sector.heldout $(data_dir)/sector.train $(model)
	./multiPred $(data_dir)/sector.train $(model)
	./multiPred $(data_dir)/sector.test $(model)
ifneq ($(p), 0)
	./multiPred $(data_dir)/sector.train $(model).p
	./multiPred $(data_dir)/sector.test $(model).p
endif
	

aloi.bin:
	./multiTrain -s $(s) -l 0.1 -m $(m) -q $(q) -g $(g) -p $(p) $(sample_opt) -h $(data_dir)/aloi/aloi.bin.heldout $(data_dir)/aloi/aloi.bin.train $(model)
	./multiPred $(data_dir)/aloi/aloi.bin.train $(model)
	./multiPred $(data_dir)/aloi/aloi.bin.test $(model)
ifneq ($(p), 0)
	./multiPred $(data_dir)/aloi/aloi.bin.train $(model).p
	./multiPred $(data_dir)/aloi/aloi.bin.test $(model).p
endif

aloi.bin2:
	./multiTrain -s $(s) -l 0.1 -m $(m) -q $(q) -g $(g) -p $(p) $(sample_opt) $(data_dir)/aloi/aloi.bin2.train $(model)
	./multiPred $(data_dir)/aloi/aloi.bin2.train $(model)
	./multiPred $(data_dir)/aloi/aloi.bin2.test $(model)
ifneq ($(p), 0)
	./multiPred $(data_dir)/aloi/aloi.bin2.train $(model).p
	./multiPred $(data_dir)/aloi/aloi.bin2.test $(model).p
endif

LSHTC1:
	./multiTrain -s $(s) -l 0.025 -c 1 -r $(r) -m $(m) -q $(q) -g $(g) -p $(p) $(sample_opt) -h $(data_dir)/LSHTC/LSHTC1/heldout.tfidf.scale.5k $(data_dir)/LSHTC/LSHTC1/train.tfidf.scale $(model)
	./multiPred $(data_dir)/LSHTC/LSHTC1/train.tfidf.scale $(model)
	./multiPred $(data_dir)/LSHTC/LSHTC1/val.tfidf.scale  $(model)
ifneq ($(p), 0)
	./multiPred $(data_dir)/LSHTC/LSHTC1/train.tfidf.scale $(model).p
	./multiPred $(data_dir)/LSHTC/LSHTC1/val.tfidf.scale  $(model).p
endif

LSHTC2:
	$(eval train_file := $(data_dir)/LSHTC/LSHTC2/wiki_large/train.tfidf.scale)
	$(eval heldout_file := $(data_dir)/LSHTC/LSHTC2/wiki_large/heldout.tfidf.scale.5k)
	$(eval test_file := $(data_dir)/LSHTC/LSHTC2/wiki_large/test.tfidf.scale.5k)
	./multiTrain -s $(s) -l 0.025 -c 1 -r $(r) -m $(m) -q $(q) -g $(g) -p $(p) $(sample_opt) -h $(heldout_file) $(train_file) $(model)
	./multiPred $(train_file) $(model)
	./multiPred $(test_file)  $(model)
ifneq ($(p), 0)
	./multiPred $(train_file) $(model).p
	./multiPred $(test_file) $(model).p
endif

