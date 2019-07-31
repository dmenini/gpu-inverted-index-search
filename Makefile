PYTHON := ~/miniconda3/envs/hdlib-env/bin/python3.6

CFLAGS := -std=c99
LFLAGS := -lm
dict_dir := ./out
D ?= 10000
SPARSITY ?= 5
N_FILES ?= 21000
QUERY ?= "scopo"
PROFILING ?= 0

ifeq ($(PROFILING), 1)
	CFLAGS += -DPROFILING
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
	${PYTHON} generate-dictionary.py ${D} ${SPARSITY} ${N_FILES} ./HDC-Language-Recognition/testing_texts ${dict_dir} 0 1

query:
	${PYTHON} query-dictionary.py ${D} ${SPARSITY} ${dict_dir} ${dict_dir}/dictionary.bin ${dict_dir} ${QUERY}

compile-c:
	gcc $(CFLAGS) -S -DD=${D} -DN_FILES=${N_FILES} query.c
	mv query.s bin/
	gcc $(CFLAGS) -o bin/query bin/query.s ${LFLAGS}
query-c: compile-c
	./bin/query ${dict_dir}/files.txt ${dict_dir}/inv_dictionary.int ${dict_dir}/query.txt ${dict_dir}

# profile-c: compile-c
# 	./bin/query ${dict_dir}/files.txt ${dict_dir}/inv_dictionary.int ${dict_dir}/query.txt ${dict_dir}
# 	mv gmon.out out/
# 	gprof -pqueryDictionary -l bin/query out/gmon.out > out/c_prof.txt
# 	# rm out/gmon.out


new-dict:
	${PYTHON} generate-dictionary.py ${D} ${SPARSITY} ${N_FILES} ./HDC-Language-Recognition/testing_texts ${dict_dir} 1 1