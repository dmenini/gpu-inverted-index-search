#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h> 

#define ARG_COUNT 			5

#ifndef D
	#define D 				10000
#endif

#ifndef N_FILES
	#define N_FILES 		21024
#endif

#define N_CHAR 	 			26
#define MAX_FILE_NAME 		100
#define MAX_TEXT_SIZE 		2000

#define INV_DICT_WIDTH		((unsigned int)(ceil(N_FILES / (float)(sizeof(int)*8))))
#define INV_DICT_REMAINDER	(N_FILES % (sizeof(int)*8))
#define INV_DICT_SIZE		(D*INV_DICT_WIDTH)
#define BAR_LEN				60

char files[N_FILES][MAX_FILE_NAME];
char dictionary[N_FILES][D+1];
char query[D];
unsigned int* match_idx;
unsigned int** inv_dictionary;
unsigned int query_ones[D];
unsigned int ones_cnt = 0;
unsigned int matches[N_FILES];
unsigned int match_cnt = 0;

int fsize(FILE* fp){
	fseek(fp, 0, SEEK_END); 
	int size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	return size;
}

void printProgress(int count, int total, char* status) {
	int filled_len = (int)ceil((double)(BAR_LEN * count / (total-1.0)));
	int percent = (int)ceil((double)(100.0 * count / (total-1.0)));
	char bar[400];
	int i = 0;
	for (; i<filled_len; i++) {
		bar[i] = '=';
	}
	for(; i<BAR_LEN; i++) {
		bar[i] = '-';
	}
	bar[i] = '\0';
	printf("[%s] %d%% %s\r", bar, percent, status);
	if (count == total-1)
		printf("\n");
}

void load_query(char file[]){

	FILE *fp = NULL;
	fp = fopen(file, "r");
	int i = 0;

	if (fp == NULL){
		printf("Error while opening %s\n", file);
	}

	for (int i=0; i<D; i++) {
		fscanf(fp, "%c", &query[i]);
	}

	fclose(fp);
}

void load_files(char file[]) {
	FILE *fp = NULL;
	fp = fopen(file, "r");

	if (fp == NULL){
		printf("Error while opening %s\n", file);
	}

	for(int i = 0; i<N_FILES; i++) {
		fscanf(fp, "%s", files[i]);
		// printf("%d: %s\n", i, files[i]);
	}
}

void load_dictionary(char file[]){

	FILE *fp = NULL;
	fp = fopen(file, "r");
	int i = 0, j = 0;

	if (fp == NULL){
		printf("Error while opening %s\n", file);
	}

	int size = fsize(fp);
	char str[MAX_FILE_NAME];
	// printf("Size: %d\n", size);

	while(ftell(fp) < size) {
	    fscanf(fp, "%[^,],%s\n", str, dictionary[i]);
		// printf("%d: ", i);
	    for(j=0; j<D; j++) {
	    	dictionary[i][j] = dictionary[i][j] - '0';
	  		// printf("%d", dictionary[i][j]);
	  	// 	if(dictionary[i][j]==1)
				// printf("%d %d\n", i, j);
		}
		// printf("\n");
	    i++;
 	}
 	printf("final i: %d\n", i);

  fclose(fp);
}

void load_inv_dictionary(char file[]){

	FILE *fp = NULL;
	fp = fopen(file, "r");

	if (fp == NULL){
		printf("Error while opening %s\n", file);
	}

	// int size = fsize(fp);
	// printf("Size: %d\n", size);
	// printf("INV_DICT_SIZE: %d\n", INV_DICT_SIZE);

	int read_cnt = 0;

	for(int j = 0; j<D; j++) {
		int i = 0;
		for( i=0; i<INV_DICT_WIDTH-1; i++){
			read_cnt = fscanf(fp, "%u ", &inv_dictionary[j][i]);
			if (read_cnt!=1)
				printf("ERROR! fscanf missed some arguments. i:\t%d\n", i);
		}
		read_cnt = fscanf(fp, "%u\n", &inv_dictionary[j][i]);
		if (read_cnt!=1)
			printf("ERROR! fscanf missed some arguments. i:\t%d\n", i);
	}

	// for (int j =0; j<100; j++) {
	// 	for (int i =0; i<INV_DICT_WIDTH; i++) {
	// 		printf("%u ", inv_dictionary[j][i]);
	// 	}
	// 	printf("\n");
	// }

	fclose(fp);
}

unsigned int findQueryOnes(char* query){
	ones_cnt = 0;
	for(unsigned int i=0; i<D; i++) {
		if(query[i] == '1')
		{
			query_ones[ones_cnt] = i;
			ones_cnt++; 
		}
	}
	return ones_cnt;
}

unsigned int queryDictionary() {

	char result = 1;
	match_cnt = 0;
	for(unsigned int j=0; j<N_FILES; j++) {
		result = 1;
		for(unsigned int i=0; i<ones_cnt; i++) {
			result = result & dictionary[j][query_ones[i]];
		}
		if (result){
			matches[match_cnt] = j;
			match_cnt++;
		}
	}
}

unsigned int queryInvertedDictionary() {

	for(unsigned int i=0; i<D; i++) {
		if(query[i] == '1') {
			for(int j=0; j<INV_DICT_WIDTH; j++){
				match_idx[j] = match_idx[j] & inv_dictionary[i][j];
			}
		}
	}

}

unsigned int findMatchOnes(){
	unsigned int match_pos = 0;
	unsigned int temp_div = 0;
	unsigned short curr_bit = 0;
	int j = 0;
	
	for(j=0; j<INV_DICT_WIDTH-1; j++) {
		temp_div = match_idx[j];
		for(int i=1; i<=(sizeof(int)*8) && temp_div>0; i++) {
			curr_bit = temp_div & 1;
			temp_div = temp_div >> 1;
			if(curr_bit==1) {
				match_pos = (j + 1)*sizeof(int)*8 - i;
				matches[match_cnt] = match_pos;
				match_cnt++;
			}
		}
	}
	// Last iteration (j already incremented) moved outside loop for remainder!=32bits
	temp_div = match_idx[j];
	for(int i=1; i<=INV_DICT_REMAINDER && temp_div>0; i++) {
		curr_bit = temp_div%2;
		temp_div = temp_div/2;
		if(curr_bit==1) {
			match_pos = j*sizeof(int)*8 + INV_DICT_REMAINDER - i;
			matches[match_cnt] = match_pos;
			match_cnt++;
		}
	}
}    

void reportQuery(char* report_f){
	FILE *fp = NULL;
	FILE *rep = fopen(report_f, "w+");

	for (int i=0; i < match_cnt; i++){
		char text[MAX_TEXT_SIZE];
		fprintf(rep, "%s\n", files[matches[i]]);
	}
	fclose(rep);
}


int main(int argc, char **argv){

	// ARGUMENTS

	if(argc != (ARG_COUNT+1)){
		printf("Requires arguments: <files input file> <dictionary input file> <query input file> <output directory> <inverted>\n");
		return 1;
	}

	char files_file[MAX_FILE_NAME]; 
	char dict_file[MAX_FILE_NAME];  
	char query_file[MAX_FILE_NAME];
	char output_dir[MAX_FILE_NAME];
	char report_file[MAX_FILE_NAME];
	int inverted = 0;

	sprintf(files_file, "%s", argv[1]);
	sprintf(dict_file, "%s", argv[2]);
	sprintf(query_file, "%s", argv[3]);
	sprintf(output_dir, "%s", argv[4]);
	sscanf(argv[5], "%d", &inverted);

	sprintf(report_file, "%s/c_query_report.txt", output_dir);
	// printf("files: %s\ndict: %s\nquery: %s\n", files_file, dict_file, query_file);

	// ALLOCATE AND INIT MEMORY FOR INV DICT

	inv_dictionary = (unsigned int**)malloc(D*sizeof(unsigned int*));
	for (int i = 0; i < D; i++) {
		inv_dictionary[i] = (unsigned int*)malloc(INV_DICT_WIDTH*sizeof(unsigned int));	
	}
	match_idx = (unsigned int*)malloc(INV_DICT_WIDTH*sizeof(unsigned int));	
	for(int j=0; j<INV_DICT_WIDTH; j++) {
		match_idx[j] = 0xFFFFFFFF;
	}

	printf("INV_DICT_WIDTH:\t%d\n", INV_DICT_WIDTH);

	// LOAD STUFF FROM FILES

	load_query(query_file);
	load_files(files_file);

	if(inverted)
		load_inv_dictionary(dict_file);
	else
		load_dictionary(dict_file);

	// QUERY DICTIONARY

	#ifdef PROFILING
		for (int i = 0; i<10000; i++) {
	#endif

			if(inverted) {
				#ifdef PROFILING
					findQueryOnes(query);
				#endif
				queryInvertedDictionary();
				findMatchOnes();
			}
			else {
				findQueryOnes(query);
				queryDictionary();
				findMatchOnes();
			}

	#ifdef PROFILING
		}
	#endif

	// FINALIZE

	reportQuery(report_file);

	free(inv_dictionary);
	free(match_idx);

	return 0;
}