PYTHON := /home/sem19f29/miniconda3/envs/hdlib-env/bin/python3.6

CFLAGS := -std=c99
dict_dir := ./out
D ?= 10000
QUERY ?= ""

dict:
	${PYTHON} generate-dictionary.py ${D} ./HDC-Language-Recognition/testing_texts ${dict_dir}

query:
	${PYTHON} query-dictionary.py ${D} ${dict_dir}/itemmemory.bin ${dict_dir}/dictionary.bin ${dict_dir} ${QUERY}

compile:
	gcc ${CFLAGS} -o bin/query -DD=${D} query.c

query-c: 
	./bin/query "" ${dict_dir}/dictionary.csv ${dict_dir}/query.txt
