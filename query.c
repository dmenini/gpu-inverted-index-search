#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARG_COUNT 		4
#define N_FILES 		21007
#define N_CHAR 	 		26
#define MAX_FILE_NAME 	100
#define MAX_TEXT_SIZE 	2000



typedef struct {
	char index[D];
	char file[MAX_FILE_NAME];
} Entry;

typedef struct {
	char bits[D];
	char letter;
} Item;

int D = 10000;
Entry dictionary[N_FILES];
Item itemMemory[N_CHAR];
char matches[N_FILES][MAX_FILE_NAME];


void load_dictionary(char file[]){

	FILE *fp = NULL;
	fp = fopen(file, "r");
	int i = 0;

	if (fp == NULL){
		printf("Error while opening %s\n", file);
		return 0;
	}

	while(feof(fp) == 0) {
        fscanf(fp, "%[^,],%s", dictionary[i].file, dictionary[i].index);
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
		return 0;
	}

    fscanf(fp, "%s", query);
    fclose(fp);

}

void load_itemmemory(char file[]) {

	FILE *fp = NULL;
	fp = fopen(file, "r");
	int i = 0;

	if (fp == NULL){
		printf("Error while opening %s\n", file);
		return 0;
	}

	while(feof(fp) == 0) {
        fscanf(fp, "%c,%s\n", itemMemory[i].letter, itemMemory[i].bits);
        i++;
    }

    fclose(fp);
}

int checkInclusion(char query[D], char index[D]){

	for (int i = 0; i < D; i++){
		if (query[i] == '1'){
			if (query[i] != index[i])
				return 0;
		}
	}
	return 1;
}

char queryDictionary(char query[D]) {

	char matches[N_FILES][NAME_SIZE];
	short int included = 0;
	int count = 0;

	for (int i = 0; i < N_FILES; i++){
		included = checkInclusion(query, dictionary[i].index);
		if (included){
			matches[count] = dictionary[i].file;
			count++;
		}
	}
    return matches;
}    

int fsize(FILE* fp){
	fseek(fp, 0, SEEK_END); 
	int size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	return size;
}

void printMatchedFiles(){
	FILE *fp = NULL;

	for (int i=0; i < N_FILES; i++){
		char text[MAX_TEXT_SIZE];
		fp = fopen(matches[i], "r");
		fgets(text, fsize(fp), fp);
		printf("File: %s\tContent: '%s'\n", matches[i], text);
		fclose(fp);
	}
}

int main(int argc, char **argv){

	// Command-line arguments:
	// D, itemmem input file, dictionary input file, query
	if(argc != ARG_COUNT){
		printf("Requires arguments: <D> <itemmem input file> <dictionary input file> <query input file>\n");
		return 0;
	}

	sscanf(argv[0], "%d", D);
	char itemmem_file[MAX_FILE_NAME] = argv[1]; 
	char dict_file[MAX_FILE_NAME] = argv[2]; 
	char query_file[MAX_FILE_NAME] = argv[3];

	char query[D];

	load_dictionary(dict_file);
	load_query(query_file, query);
	// encoder = load_itemmemory(itemmem_file);

    matches = queryDictionary(query);
    printMatchedFiles();

    return 1;
}