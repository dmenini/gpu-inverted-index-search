# Use PROFILING only to profile, not to run the app. If you run the program with it, the query will be wrong.

PYTHON := ~/miniconda3/envs/hdlib-env/bin/python3.6

CFLAGS := -std=c99 -g -pg
LFLAGS := -lm
NVCCFLAGS := -O3 --use_fast_math -g -G -std=c++11 -Xcompiler '-fopenmp' --gpu-architecture=compute_61 --compiler-options -fPIC --linker-options --no-undefined 
dict_dir := ./out
D ?= 10000
SPARSITY ?= 5
QUERY ?= "scopo"
PROFILING ?= 0
DUP ?= 1

dict_file 		= dictionary_$(DUP).bin
inv_dict_file = inv_dictionary_$(DUP).int
files_file		= files_$(DUP).txt
prof_dir			= prof
prof_file			= prof_$(DUP).txt
N_FILES 	?= $(shell echo $(DUP)*21024 | bc)

ifeq ($(PROFILING), 1)
	CFLAGS += -DPROFILING
	NVCCFLAGS += -DPROFILING
endif

clean:
	rm -rf bin
	rm -rf out

setup: clean
	rm -rf hdlib
	rm -rf HDC-Language-Recognition
	git clone https://github.com/bigcola317/hdlib.git
	git clone https://github.com/abbas-rahimi/HDC-Language-Recognition.git

invert-dict:
	${PYTHON} src/dictionary.py ${D} ${SPARSITY} ${N_FILES} ./HDC-Language-Recognition/testing_texts ${dict_dir} 0 0 $(DUP)

dict:
	${PYTHON} src/dictionary.py ${D} ${SPARSITY} ${N_FILES} ./HDC-Language-Recognition/testing_texts ${dict_dir} 0 1 $(DUP)

new-dict:
	${PYTHON} src/dictionary.py ${D} ${SPARSITY} ${N_FILES} ./HDC-Language-Recognition/testing_texts ${dict_dir} 1 1 $(DUP)

dup:
	rm bin/dup.s
	rm bin/dup.out
	gcc -S -std=c99 src/dup.c
	mv dup.s bin/
	gcc -o bin/dup.out bin/dup.s ${LFLAGS}
	./bin/dup.out 4 $(dict_dir)/inv_dictionary_1.int $(dict_dir)/inv_dictionary_4.int $(dict_dir)/files_1.txt $(dict_dir)/files_4.txt ${N_FILES} 10000
	@echo "******************************************************************************************************************************************************"
	@echo "                                                              SUCCESS 4"
	@echo "******************************************************************************************************************************************************"	
	./bin/dup.out 8 $(dict_dir)/inv_dictionary_1.int $(dict_dir)/inv_dictionary_8.int $(dict_dir)/files_1.txt $(dict_dir)/files_8.txt ${N_FILES} 10000
	@echo "******************************************************************************************************************************************************"
	@echo "                                                              SUCCESS 8"
	@echo "******************************************************************************************************************************************************"
	./bin/dup.out 16 $(dict_dir)/inv_dictionary_1.int $(dict_dir)/inv_dictionary_16.int $(dict_dir)/files_1.txt $(dict_dir)/files_16.txt ${N_FILES} 10000
	@echo "******************************************************************************************************************************************************"
	@echo "                                                              SUCCESS 16"
	@echo "******************************************************************************************************************************************************"
	./bin/dup.out 32 $(dict_dir)/inv_dictionary_1.int $(dict_dir)/inv_dictionary_32.int $(dict_dir)/files_1.txt $(dict_dir)/files_32.txt ${N_FILES} 10000
	@echo "******************************************************************************************************************************************************"
	@echo "                                                              SUCCESS 32"
	@echo "******************************************************************************************************************************************************"
	./bin/dup.out 64 $(dict_dir)/inv_dictionary_1.int $(dict_dir)/inv_dictionary_64.int $(dict_dir)/files_1.txt $(dict_dir)/files_64.txt ${N_FILES} 10000
	@echo "******************************************************************************************************************************************************"
	@echo "                                                              SUCCESS 64"
	@echo "******************************************************************************************************************************************************"
	./bin/dup.out 128 $(dict_dir)/inv_dictionary_1.int $(dict_dir)/inv_dictionary_128.int $(dict_dir)/files_1.txt $(dict_dir)/files_128.txt ${N_FILES} 10000
	@echo "******************************************************************************************************************************************************"
	@echo "                                                             SUCCESS 128"
	@echo "******************************************************************************************************************************************************"

compile-c:
	gcc $(CFLAGS) -S -DD=${D} -DN_FILES=${N_FILES} src/query.c
	mv query.s bin/
	gcc $(CFLAGS) -o bin/query bin/query.s ${LFLAGS}

compile-cu:
	nvcc -o bin/query-cu.out src/query.cu -DD=${D} -DN_FILES=${N_FILES} $(NVCCFLAGS) 
	nvcc -o bin/query-cu-half.out src/query_half.cu -DD=${D} -DN_FILES=${N_FILES} $(NVCCFLAGS)

query-py:
	${PYTHON} src/query.py ${D} ${SPARSITY} ${dict_dir} ${dict_dir}/$(dict_file) ${dict_dir} ${QUERY}

query-basic-c: compile-c
	./bin/query ${dict_dir}/$(files_file) ${dict_dir}/dictionary_1.csv ${dict_dir}/query.txt ${dict_dir} 0

query-c: compile-c
	./bin/query ${dict_dir}/$(files_file) ${dict_dir}/$(inv_dict_file) ${dict_dir}/query.txt ${dict_dir} 1

query-cu: compile-cu
	./bin/query-cu.out ./out/$(files_file) ./out/$(inv_dict_file) ./out/query.txt ./out

queries: query-c query-cu-

profile-c: 
	$(MAKE) query-c
	mv gmon.out out/
	gprof -b bin/query out/gmon.out
	gprof -b bin/query out/gmon.out > ${prof_dir}/c_$(prof_file)
	rm out/gmon.out
	# $(MAKE) query-basic-c
	# mv gmon.out out/
	# gprof -b bin/query out/gmon.out
	# gprof -b bin/query out/gmon.out > ${prof_dir}/c_basic_$(prof_file)
	# rm out/gmon.out

profile-cu: compile-cu # compile-c profile-c
	nvprof ./bin/query-cu.out ./out/$(files_file) ./out/$(inv_dict_file) ./out/query.txt ./out > ${prof_dir}/cu_$(prof_file)
	@echo "******************************************************************************************************************************************************"
	@echo "******************************************************************************************************************************************************"
	nvprof ./bin/query-cu-half.out ./out/$(files_file) ./out/$(inv_dict_file) ./out/query.txt ./out > ${prof_dir}/cu_half_$(prof_file)

profile: profile-c profile-cu

nvvp: compile-cu
	nvprof --analysis-metrics -f -o out/cuda_nvprof_rep.nvvp ./bin/query-cu.out ./out/$(files_file) ./out/$(inv_dict_file) ./out/query.txt ./out
	nvvp out/cuda_nvprof_rep.nvvp

debug: compile-cu
	rm ../.cuda-gdbinit
	echo "file bin/query-cu.out  " > ../.cuda-gdbinit
	echo "break queryKernel" >> ../.cuda-gdbinit 
	echo "run ./out/$(files_file) ./out/$(inv_dict_file) ./out/query.txt ./out" >> ../.cuda-gdbinit
	echo "define d1" >> ../.cuda-gdbinit
	echo "print i" >> ../.cuda-gdbinit
	echo "print threadIdx.x" >> ../.cuda-gdbinit
	echo "print match_idx_s[threadIdx.x]" >> ../.cuda-gdbinit
	echo "print *(inv_dictionary + query_ones[i]*inv_dict_width_pad + index)" >> ../.cuda-gdbinit
	echo "print query_ones[i]" >> ../.cuda-gdbinit
	echo "continue" >> ../.cuda-gdbinit
	echo "end" >> ../.cuda-gdbinit
	cuda-gdb