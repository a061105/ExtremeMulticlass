all: train predict
	
FLAG= -DUSING_HASHVEC #-DMULTISELECT
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
	./multiTrain -s $(s) -l 0.05 -m $(m) -q $(q) -p $(p) $(sample_opt) -h $(data_dir)/aloi/aloi.bin.heldout $(data_dir)/aloi/aloi.bin.train $(model)
	./multiPred $(data_dir)/aloi/aloi.bin.train $(model)
	./multiPred $(data_dir)/aloi/aloi.bin.test $(model)
ifneq ($(p), 0)
	./multiPred $(data_dir)/aloi/aloi.bin.train $(model).p
	./multiPred $(data_dir)/aloi/aloi.bin.test $(model).p
endif

Dmoz:
	$(eval train_file := $(data_dir)/ODP/Dmoz.train)
	$(eval heldout_file := $(data_dir)/ODP/Dmoz.heldout.10k)
	$(eval test_file := $(data_dir)/ODP/Dmoz.test)
	./multiTrain -s $(s) -l $(l) -c 1 -r $(r) -m $(m) -q $(q) -g $(g) -p $(p) $(sample_opt) -h $(heldout_file) $(train_file) $(model)
	./multiPred $(train_file) $(model)
	./multiPred $(test_file)  $(model)
ifneq ($(p), 0)
	./multiPred $(train_file) $(model).p
	./multiPred $(test_file) $(model).p
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
	$(eval train_file := $(data_dir)/LSHTC/LSHTC1/LSHTC1.train)
	$(eval heldout_file := $(data_dir)/LSHTC/LSHTC1/LSHTC1.heldout.5k)
	$(eval test_file := $(data_dir)/LSHTC/LSHTC1/LSHTC1.test.5k)
	./multiTrain -s $(s) -l $(l) -c 1 -r 15 -m $(m) -q $(q) -p $(p) $(sample_opt) -h $(heldout_file) $(train_file) $(model)
	#./multiPred $(train_file) $(model)
	./multiPred $(test_file)  $(model)
ifneq ($(p), 0)
	#./multiPred $(train_file) $(model).p
	./multiPred $(test_file) $(model).p
endif

LSHTC2:
	$(eval train_file := $(data_dir)/LSHTC/LSHTC2/wiki_large/train.tfidf.scale)
	$(eval heldout_file := $(data_dir)/LSHTC/LSHTC2/wiki_large/heldout.tfidf.scale.5k)
	$(eval test_file := $(data_dir)/LSHTC/LSHTC2/wiki_large/test.tfidf.scale.5k)
	./multiTrain -s $(s) -l $(l) -c 1 -m $(m) -p $(p) $(sample_opt) -h $(heldout_file) $(train_file) $(model)
	./multiPred $(train_file) $(model)
	./multiPred $(test_file)  $(model)
ifneq ($(p), 0)
	./multiPred $(train_file) $(model).p
	./multiPred $(test_file) $(model).p
endif

imgNet:	
	$(eval train_file := $(data_dir)/imagenet/imgNet.train)
	$(eval heldout_file := $(data_dir)/imagenet/imgNet.heldout)
	$(eval test_file := $(data_dir)/imagenet/imgNet.test)
	./multiTrain -s $(s) -l $(l) -r 10 -c 1 -m $(m) -q $(q) -p $(p) $(sample_opt) -h $(heldout_file) $(train_file) $(model)
	./multiPred $(train_file) $(model)
	./multiPred $(test_file)  $(model)
ifneq ($(p), 0)
	./multiPred $(train_file) $(model).p
	./multiPred $(test_file) $(model).p
endif

bibtex:
	$(eval train_file := $(data_dir)/multilabel/bibtex.train)
	$(eval heldout_file := $(data_dir)/multilabel/bibtex.heldout)
	$(eval test_file := $(data_dir)/multilabel/bibtex.test)
	./multiTrain -s $(s) -l $(l) -c 1 -m $(m) -q $(q) -p $(p) $(sample_opt) -h $(heldout_file) $(train_file) $(model)
	#./multiPred $(train_file) $(model)
	./multiPred $(test_file)  $(model)
ifneq ($(p), 0)
	#./multiPred $(train_file) $(model).p
	./multiPred $(test_file) $(model).p
endif	
