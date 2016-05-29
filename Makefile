all: multiTrain multiTrainHash multiPred
	
multiTrain:
	g++ -fopenmp -std=c++11 -O3 -o multiTrain multiTrain.cpp
multiTrainHash:	
	g++ -fopenmp -std=c++11 -O3 -o multiTrainHash multiTrain.cpp -DUSING_HASHVEC
	
multiPred:
	g++ -fopenmp -std=c++11 -O3 -o multiPred multiPred.cpp

clean:
	rm -f multiTrain
	rm -f multiTrainHash
	rm -f multiPred

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
data_dir=/scratch/cluster/xrhuang/data
train_file=
heldout_file=
test_file=

.SECONDEXPANSION:

#multilabel datasets
LSHTCwiki: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test

rcv1_regions:  examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda=0.1 split_up_rate=3 

bibtex: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test sample_option=-u early_terminate=10

Eur-Lex: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda=0.001 early_terminate=10 

#multiclass datasets
sector: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda=0.1

aloi.bin: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda=0.01

Dmoz: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda=0.1 split_up_rate=3

LSHTC1: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda=0.01 split_up_rate=3 early_terminate=10

imageNet: examples/$$@/	
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda=0.1 split_up_rate=3

train_without_hash: multiTrain multiPred $(train_file) $(heldout_file) $(test_file)
	./multiTrain -c $(cost) -l $(lambda) -s $(solver) -r $(speed_up_rate) -e $(early_terminate) -m $(max_iter) -q $(split_up_rate) -g $(max_select) -p $(post_train_iter) $(sample_option) -h $(heldout_file) $(train_file) $(output_model)
	@echo "testing model before post solve"
	./multiPred $(test_file) $(output_model)
ifneq ($(p), 0)
	@echo "testing model after post solve"
	./multiPred $(test_file) $(output_model).p
endif

train_with_hash: multiTrainHash multiPred $(train_file) $(heldout_file) $(test_file)
	./multiTrainHash -c $(cost) -l $(lambda) -s $(solver) -r $(speed_up_rate) -e $(early_terminate) -m $(max_iter) -q $(split_up_rate) -g $(max_select) -p $(post_train_iter) $(sample_option) -h $(heldout_file) $(train_file) $(output_model)
	@echo "testing model before post solve"
	./multiPred $(test_file) $(output_model)
ifneq ($(p), 0)
	@echo "testing model after post solve"
	./multiPred $(test_file) $(output_model).p
endif

examples/%:
	make construct -C examples/ dataset=$(notdir $@)

