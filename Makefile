all: train predict
	
train:
	g++ -fopenmp -std=c++11 -O3 -o multiTrainHash multiTrain.cpp -DUSING_HASHVEC
	g++ -fopenmp -std=c++11 -O3 -o multiTrain multiTrain.cpp
	
predict:
	g++ -fopenmp -std=c++11 -O3 -o multiPred multiPred.cpp

#parameters
solver=1
lambda=1.0
cost=1.0
speed_up_rate=-1.0
split_up_rate=1
max_iter=200
sample_option=
max_select=-1
post_train_iter=200
early_terminate=3

output_model=model
data_dir=./examples#/scratch/cluster/xrhuang/data
train_file=
heldout_file=
test_file=

#multilabel datasets
LSHTCwiki:
	$(eval train_file := $(data_dir)/multilabel/$@.train)
	$(eval heldout_file := $(data_dir)/multilabel/$@.heldout)
	$(eval test_file := $(data_dir)/multilabel/$@.test)
	make train_with_hash train_file=$(train_file) heldout_file=$(heldout_file) test_file=$(test_file) 

rcv1_regions:
	$(eval train_file := $(data_dir)/multilabel/$@.train)
	$(eval heldout_file := $(data_dir)/multilabel/$@.heldout)
	$(eval test_file := $(data_dir)/multilabel/$@.test)
	make train_without_hash train_file=$(train_file) heldout_file=$(heldout_file) test_file=$(test_file) lambda=0.1 split_up_rate=3 

bibtex:
	$(eval train_file := $(data_dir)/multilabel/$@.train)
	$(eval heldout_file := $(data_dir)/multilabel/$@.heldout)
	$(eval test_file := $(data_dir)/multilabel/$@.test)
	make train_without_hash train_file=$(train_file) heldout_file=$(heldout_file) test_file=$(test_file) sample_option=-u 

Eur-Lex:
	$(eval train_file := $(data_dir)/multilabel/$@.train)
	$(eval heldout_file := $(data_dir)/multilabel/$@.heldout)
	$(eval test_file := $(data_dir)/multilabel/$@.test)
	make train_with_hash train_file=$(train_file) heldout_file=$(heldout_file) test_file=$(test_file) lambda=0.001 early_terminate=10 

#multiclass datasets
sector:
	$(eval train_file := $(data_dir)/$@/$@.train)
	$(eval heldout_file := $(data_dir)/$@/$@.heldout)
	$(eval test_file := $(data_dir)/$@/$@.test)
	make train_without_hash train_file=$(train_file) heldout_file=$(heldout_file) test_file=$(test_file) lambda=0.1

aloi.bin:
	$(eval train_file := $(data_dir)/$@/$@.train)
	$(eval heldout_file := $(data_dir)/$@/$@.heldout)
	$(eval test_file := $(data_dir)/$@/$@.test)
	make train_with_hash train_file=$(train_file) heldout_file=$(heldout_file) test_file=$(test_file) lambda=0.01

Dmoz:
	$(eval train_file := $(data_dir)/Dmoz/Dmoz.train)
	$(eval heldout_file := $(data_dir)/Dmoz/Dmoz.heldout)
	$(eval test_file := $(data_dir)/Dmoz/Dmoz.test)
	make train_with_hash train_file=$(train_file) heldout_file=$(heldout_file) test_file=$(test_file)

LSHTC1:
	$(eval train_file := $(data_dir)/LSHTC/LSHTC1/LSHTC1.train)
	$(eval heldout_file := $(data_dir)/LSHTC/LSHTC1/LSHTC1.heldout)
	$(eval test_file := $(data_dir)/LSHTC/LSHTC1/LSHTC1.test)
	make train_with_hash train_file=$(train_file) heldout_file=$(heldout_file) test_file=$(test_file) lambda=0.01 split_up_rate=3 early_terminate=10

imgNet:	
	$(eval train_file := $(data_dir)/imagenet/imgNet.train)
	$(eval heldout_file := $(data_dir)/imagenet/imgNet.heldout)
	$(eval test_file := $(data_dir)/imagenet/imgNet.test)
	make train_with_hash train_file=$(train_file) heldout_file=$(heldout_file) test_file=$(test_file)



train_without_hash: $(train_file) $(heldout_file) $(test_file)
	./multiTrain -c $(cost) -l $(lambda) -s $(solver) -r $(speed_up_rate) -e $(early_terminate) -m $(max_iter) -q $(split_up_rate) -g $(max_select) -p $(post_train_iter) $(sample_option) -h $(heldout_file) $(train_file) $(output_model)
	#testing model before post solve
	./multiPred $(test_file) $(output_model)
ifneq ($(p), 0)
	#testing model after post solve
	./multiPred $(test_file) $(output_model).p
endif

train_with_hash: $(train_file) $(heldout_file) $(test_file)
	./multiTrainHash -c $(cost) -l $(lambda) -s $(solver) -r $(speed_up_rate) -e $(early_terminate) -m $(max_iter) -q $(split_up_rate) -g $(max_select) -p $(post_train_iter) $(sample_option) -h $(heldout_file) $(train_file) $(output_model)
	#testing model before post solve
	./multiPred $(test_file) $(output_model)
ifneq ($(p), 0)
	#testing model after post solve
	./multiPred $(test_file) $(output_model).p
endif
