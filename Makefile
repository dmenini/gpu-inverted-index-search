PYTHON := /home/sem19f29/miniconda3/envs/hdlib-env/bin/python3.6

CFLAGS := -std=c99
dict_dir := ./out
D ?= 10000
N_FILES ?= 21007
QUERY ?= "scopo"

clean:
	rm -rf bin
	rm -rf out

setup: clean
	rm -rf hdlib
	rm -rf HDC-Language-Recognition
	git clone https://github.com/bigcola317/hdlib.git
	git clone https://github.com/abbas-rahimi/HDC-Language-Recognition.git

dict:
	${PYTHON} generate-dictionary.py ${D} ${N_FILES} ./HDC-Language-Recognition/testing_texts ${dict_dir} 0 1

query:
	${PYTHON} query-dictionary.py ${D} ${dict_dir} ${dict_dir}/dictionary.bin ${dict_dir} ${QUERY}

compile:
	gcc ${CFLAGS} -o bin/query -DD=${D} -DN_FILES=${N_FILES} query.c

inv-query-c: compile
	./bin/query ${dict_dir}/files.txt ${dict_dir}/inv_dictionary.int ${dict_dir}/query.txt

new-dict:
	${PYTHON} generate-dictionary.py ${D} ${N_FILES} ./HDC-Language-Recognition/testing_texts ${dict_dir} 1