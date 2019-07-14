#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARG_COUNT 			3

#ifndef D
	#define D 							10000
#endif

#define N_FILES 				21007
#define N_CHAR 	 				26
#define MAX_FILE_NAME 	100
#define MAX_TEXT_SIZE 	2000

typedef struct {
	char index[D+1];
	char file[MAX_FILE_NAME];
} Entry;

typedef struct {
	char bits[D+1];
	char letter;
} Item;

Entry dictionary[N_FILES];
Item itemMemory[N_CHAR];
char matches[N_FILES][MAX_FILE_NAME];


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
    fscanf(fp, "%[^,],%s\n", dictionary[i].file, dictionary[i].index);
  	// printf("i: %d\nfile: %s\nindex: %s\npos: %d\n", i, dictionary[i].file, dictionary[i].index, ftell(fp));
    i++;
  }

  fclose(fp);
}

void load_query(char file[], char query[]){

	FILE *fp = NULL;
	fp = fopen(file, "r");
	int i = 0;

	if (fp == NULL){
		printf("Error while opening %s\n", file);
	}

  fscanf(fp, "%s", query);
  // printf("%s\n", query);
  fclose(fp);

}

// void load_itemmemory(char file[]) {

// 	FILE *fp = NULL;
// 	fp = fopen(file, "r");
// 	int i = 0;

// 	if (fp == NULL){
// 		printf("Error while opening %s\n", file);
// 	}

// 	while(feof(fp) == 0) {
//         fscanf(fp, "%c,%s\n", itemMemory[i].letter, itemMemory[i].bits);
//         i++;
//     }

//     fclose(fp);
// }

int checkInclusion(char query[], char index[]){

	for (int i = 0; i < D; i++){
		if (query[i] == '1'){
			if (query[i] != index[i])
				return 0;
		}
	}
	return 1;
}

void queryDictionary(char query[]) {

	short int included = 0;
	int count = 0;

	for (int i = 0; i < N_FILES; i++){
		included = checkInclusion(query, dictionary[i].index);
		if (included){
			sprintf(matches[count], "%s", dictionary[i].file);
			count++;
		}
	}
	for(int i=count; i < N_FILES; i++){
		sprintf(matches[i], "%s", "");
	}
}    

void printMatchedFiles(){
	FILE *fp = NULL;

	for (int i=0; i < N_FILES; i++){
		char text[MAX_TEXT_SIZE];
		if (strcmp(matches[i], "")) {
			fp = fopen(matches[i], "r");
			if (fp == NULL){
				printf("Error while opening '%s'\n", matches[i]);
			}
			fgets(text, fsize(fp), fp);
			printf("File: %s\tContent: '%s'\n", matches[i], text);
			fclose(fp);
		}
	}
}

int main(int argc, char **argv){

	// Command-line arguments:
	// D, itemmem input file, dictionary input file, query
	if(argc != (ARG_COUNT+1)){
		printf("Requires arguments: <itemmem input file> <dictionary input file> <query input file>\n");
		return 1;
	}

	char itemmem_file[MAX_FILE_NAME]; 
	char dict_file[MAX_FILE_NAME];  
	char query_file[MAX_FILE_NAME];

	sprintf(itemmem_file, "%s", argv[1]);
	sprintf(dict_file, "%s", argv[2]);
	sprintf(query_file, "%s", argv[3]);

	printf("itemmem: %s\ndict: %s\nquery: %s\n", itemmem_file, dict_file, query_file);

	char query[D+1];

	load_dictionary(dict_file);
	load_query(query_file, query);
	// encoder = load_itemmemory(itemmem_file);

  queryDictionary(query);
  printMatchedFiles();

  return 0;
}