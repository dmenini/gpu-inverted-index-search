PYTHON := python3.6

dict_dir := "./out"
D ?= 10000
QUERY ?= ""

dict:
	${PYTHON} generate-dictionary.py ${D} ./HDC-Language-Recognition/testing_texts ${dict_dir}

query:
	${PYTHON} query-dictionary.py ${D} ${dict_dir}/itemmemory.bin ${dict_dir}/dictionary.bin ${dict_dir} ${QUERY}

query-c: query
	 
