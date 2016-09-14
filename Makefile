all: multiTrain multiTrainHash multiPred
	
#.PHONY: multiTrain multiTrainHash multiPred

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

output_model=model
data_dir=/scratch/cluster/xrhuang/data
train_file=
heldout_file=
test_file=
misc=

.SECONDEXPANSION:

#multilabel datasets
LSHTCwiki_original: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda="-l 0.01" output_model='LSHTCwiki.model" misc="-d"

rcv1_regions:  examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test split_up_rate="-q 3" 

bibtex: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test sample_option="-u" early_terminate="-e 10" speed_up_rate="-r 1" lambda="-l 1"

Eur-Lex: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda="-l 0.001" early_terminate="-e 10"

#multiclass datasets
sector: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_without_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test

aloi.bin: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda="-l 0.01"

Dmoz: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test split_up_rate="-q 3"

LSHTC1: examples/$$@/
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test lambda="-l 0.01" split_up_rate="-q 3" early_terminate="-e 3"

imageNet: examples/$$@/	
	$(eval base := examples/$@/$@)
	make train_with_hash train_file=$(base).train heldout_file=$(base).heldout test_file=$(base).test split_up_rate="-q 3"

train_without_hash: multiTrain multiPred $(train_file) $(heldout_file) $(test_file)
	./multiTrain $(cost) $(lambda) $(solver) $(speed_up_rate) $(early_terminate) $(max_iter) $(split_up_rate) $(max_select) $(post_train_iter) $(sample_option) $(misc) -h $(heldout_file) $(train_file) $(output_model)
	@echo "testing model before post solve"
	./multiPred $(test_file) $(output_model) 
ifneq ($(p), 0)
	@echo "testing model after post solve"
	./multiPred $(test_file) $(output_model).p
endif

train_with_hash: multiTrainHash multiPred $(train_file) $(heldout_file) $(test_file)
	./multiTrainHash $(cost) $(lambda) $(solver) $(speed_up_rate) $(early_terminate) $(max_iter) $(split_up_rate) $(max_select) $(post_train_iter) $(sample_option) $(misc) -h $(heldout_file) $(train_file) $(output_model)
	@echo "testing model before post solve"
	./multiPred $(test_file) $(output_model)
ifneq ($(p), 0)
	@echo "testing model after post solve"
	./multiPred $(test_file) $(output_model).p
endif

examples/%:
	make construct -C examples/ dataset=$(notdir $@)

