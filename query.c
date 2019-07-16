#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 

#define ARG_COUNT 			3

#ifndef D
	#define D 				10000
#endif

#ifndef N_FILES
	#define N_FILES 		21007
#endif

#define N_CHAR 	 			26
#define MAX_FILE_NAME 		100
#define MAX_TEXT_SIZE 		2000

#define INV_DICT_WIDTH		((N_FILES / (sizeof(int)*8)) + 1)
#define INV_DICT_REMAINDER	(N_FILES % (sizeof(int)*8))
#define INV_DICT_SIZE		(D*INV_DICT_WIDTH)

char files[N_FILES][MAX_FILE_NAME];
char dictionary[N_FILES][D+1];
unsigned int inv_dictionary[D][INV_DICT_WIDTH];
char query[D];
char matches[N_FILES][MAX_FILE_NAME];
unsigned int match_idx[INV_DICT_WIDTH];
unsigned int match_cnt = 0;

int fsize(FILE* fp){
	fseek(fp, 0, SEEK_END); 
	int size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	return size;
}

void load_dictionary(char file[]){

	FILE *fp = NULL;
	fp = fopen(file, "r");
	int i = 0;

	if (fp == NULL){
		printf("Error while opening %s\n", file);
	}

	int size = fsize(fp);
	// printf("Size: %d\n", size);

	while(ftell(fp) < size) {
    fscanf(fp, "%[^,],%s\n", files[i], dictionary[i]);
  	// printf("i: %d\nfile: %s\nindex: %s\npos: %d\n", i, dictionary[i].file, dictionary[i].index, ftell(fp));
    i++;
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

void load_inv_dictionary(char file[]){

	FILE *fp = NULL;
	fp = fopen(file, "r");

	if (fp == NULL){
		printf("Error while opening %s\n", file);
	}

	// int size = fsize(fp);
	// printf("Size: %d\n", size);
	// printf("INV_DICT_SIZE: %d\n", INV_DICT_SIZE);

	for(int j = 0; j<D; j++) {
		int i = 0;
		for( i=0; i<INV_DICT_WIDTH-1; i++){
			fscanf(fp, "%u ", &inv_dictionary[j][i]);
		}
		fscanf(fp, "%u\n", &inv_dictionary[j][i]);
	}

	// for (int j =0; j<100; j++) {
	// 	for (int i =0; i<INV_DICT_WIDTH; i++) {
	// 		printf("%u ", inv_dictionary[j][i]);
	// 	}
	// 	printf("\n");
	// }

	fclose(fp);
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

	// printf("Loaded query: ");

	// for (int i=0; i<D; i++) {
	// 	printf("%c", query[i]);
	// }

	// printf("\n");

	fclose(fp);

}

unsigned int queryDictionary() {
	unsigned int file_index = 0;
	unsigned int temp_div = 0;
	unsigned short curr_bit = 0;

	for(int j=0; j<INV_DICT_WIDTH; j++) {
		match_idx[j] = 0xFFFFFFFF;
	}

//////////////////////////     COMPUTATION STARTS HERE    //////////////////////////////////////////////

	for(int i = 0; i<D; i++) {
		if (query[i] == '1') {
			// printf("%u: ", i);
			for(int j=0; j<INV_DICT_WIDTH; j++){
				// printf("match[%u]: %u\tdict[%u]: %u\t", j, match_idx[j], j, inv_dictionary[i][j]);
				match_idx[j] = match_idx[j] & inv_dictionary[i][j];
				// printf("res[%u]: %u\n", j, match_idx[j]);
			}
		}
	}

	// printf("High bits: ");
	int j = 0;
	for(j=0; j<INV_DICT_WIDTH-1; j++) {
		temp_div = match_idx[j];
		for(int i=1; i<=(sizeof(int)*8) || temp_div>0; i++) {
			curr_bit = temp_div%2;
			temp_div = temp_div/2;
			if(curr_bit==1) {
				// printf("%u", i);
				file_index = (j + 1)*sizeof(int)*8 - i;
				// printf("(%u) ", file_index);
				strcpy(matches[match_cnt], files[file_index]);
				match_cnt++;
			}
		}
	}
	// Last iteration (j already incremented) moved outside loop for remainder!=32bits
	temp_div = match_idx[j];
	for(int i=1; i<=INV_DICT_REMAINDER || temp_div>0; i++) {
		curr_bit = temp_div%2;
		temp_div = temp_div/2;
		if(curr_bit==1) {
			// printf("%u", i);
			file_index = j*sizeof(int)*8 + INV_DICT_REMAINDER - i;
			// printf("(%u) ", file_index);
			strcpy(matches[match_cnt], files[file_index]);
			match_cnt++;
		}
	}

//////////////////////////    COMPUTATION ENDS HERE    //////////////////////////////////////////////

}    

void printMatchedFiles(){
	FILE *fp = NULL;

	for (int i=0; i < match_cnt; i++){
		char text[MAX_TEXT_SIZE];
		fp = fopen(matches[i], "r");
		if (fp == NULL){
			printf("Error while opening '%s'\n", matches[i]);
		}
		fgets(text, fsize(fp), fp);
		printf("File: %s\tContent: '%s'\n", matches[i], text);
		fclose(fp);
	}
}

int main(int argc, char **argv){

	// Command-line arguments:
	// D, itemmem input file, dictionary input file, query
	if(argc != (ARG_COUNT+1)){
		printf("Requires arguments: <files input file> <dictionary input file> <query input file>\n");
		return 1;
	}

	char files_file[MAX_FILE_NAME]; 
	char dict_file[MAX_FILE_NAME];  
	char query_file[MAX_FILE_NAME];

	sprintf(files_file, "%s", argv[1]);
	sprintf(dict_file, "%s", argv[2]);
	sprintf(query_file, "%s", argv[3]);

	printf("files: %s\ndict: %s\nquery: %s\n", files_file, dict_file, query_file);

	load_files(files_file);
	load_inv_dictionary(dict_file);
	load_query(query_file);

	queryDictionary(query);
	printMatchedFiles();

	return 0;
}