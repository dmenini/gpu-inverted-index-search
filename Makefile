# Use PROFILING only to profile, not to run the app. If you run the program with it, the query will be wrong.

PYTHON := ~/miniconda3/envs/hdlib-env/bin/python3.6

CFLAGS := -std=c99 -g -pg
LFLAGS := -lm
NVCCFLAGS := -O3 --use_fast_math -g -G -std=c++11 -Xcompiler '-fopenmp' --gpu-architecture=compute_61 --compiler-options -fPIC --linker-options --no-undefined
dict_dir := ./out
D ?= 10000
SPARSITY ?= 5
N_FILES ?= 21000
QUERY ?= "scopo"
PROFILING ?= 0
DUP ?= 0

ifeq ($(DUP), 8)
	dict_file 		= dictionary_$(DUP).bin
	inv_dict_file = inv_dictionary_$(DUP).int
	files_file		= files_$(DUP).txt
	N_FILES 			= 168000
else
	dict_file 		= dictionary.bin
	inv_dict_file = inv_dictionary.int
	files_file		= files.txt
endif

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

dict:
	${PYTHON} src/generate-dictionary.py ${D} ${SPARSITY} ${N_FILES} ./HDC-Language-Recognition/testing_texts ${dict_dir} 0 1

query:
	${PYTHON} src/query-dictionary.py ${D} ${SPARSITY} ${dict_dir} ${dict_dir}/$(dict_file) ${dict_dir} ${QUERY}

compile-c:
	gcc $(CFLAGS) -S -DD=${D} -DN_FILES=${N_FILES} src/query.c
	mv query.s bin/
	gcc $(CFLAGS) -o bin/query bin/query.s ${LFLAGS}

query-c: compile-c
	./bin/query ${dict_dir}/$(files_file) ${dict_dir}/$(inv_dict_file) ${dict_dir}/query.txt ${dict_dir}

compile-cu:
	nvcc -o bin/query-cu.out src/query.cu -DD=${D} -DN_FILES=${N_FILES} $(NVCCFLAGS)

query-cu: compile-cu
	./bin/query-cu.out ./out/$(files_file) ./out/$(inv_dict_file) ./out/query.txt ./out

profile-c: query-c
	mv gmon.out out/
	gprof -b bin/query out/gmon.out
	rm out/gmon.out

profile: compile-c compile-cu # profile-c
	# nvprof --analysis-metrics -f -o out/cuda_nvprof_rep.nvvp ./bin/query-cu.out ./out/files.txt ./out/inv_dictionary.int ./out/query.txt ./out
	nvprof ./bin/query-cu.out ./out/$(files_file) ./out/$(inv_dict_file) ./out/query.txt ./out

new-dict:
	${PYTHON} src/generate-dictionary.py ${D} ${SPARSITY} ${N_FILESq} ./HDC-Language-Recognition/testing_texts ${dict_dir} 1 1

debug: compile-cu
	rm ../.cuda-gdbinit
	echo "file bin/query-cu.out  " > ../.cuda-gdbinit
	echo "break queryKernel" >> ../.cuda-gdbinit 
	echo "run ./out/$(files_file) ./out/$(inv_dict_file) ./out/query.txt ./out" >> ../.cuda-gdbinit
	echo "define d1" >> ../.cuda-gdbinit
	echo "print {index, match_idx}" >> ../.cuda-gdbinit
	echo "continue" >> ../.cuda-gdbinit
	echo "end" >> ../.cuda-gdbinit
	cuda-gdb