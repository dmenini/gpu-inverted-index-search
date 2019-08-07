#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ARG_COUNT 7
#define MAX_FILE_NAME 100
#define BAR_LEN				60

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

int duplicate(int DUP, char* i_filename, char* o_filename, char* i_txt_filename, char* o_txt_filename, unsigned int num_cols, unsigned int num_rows){
	FILE* i_f = fopen(i_filename, "r");
	FILE* o_f = fopen(o_filename, "w");
	FILE* i_txt_f = fopen(i_txt_filename, "r");
	FILE* o_txt_f = fopen(o_txt_filename, "w");

	// CHECK OPENED FILES
	
	if(i_f == NULL) {
		printf("Failed to open input file %s\n", i_filename);
		return 1;
	}
	if(o_f == NULL) {
		printf("Failed to open output file %s\n", o_filename);
		return 1;
	}
	if(i_txt_f == NULL) {
		printf("Failed to open input file %s\n", i_txt_filename);
		return 1;
	}
	if(o_txt_f == NULL) {
		printf("Failed to open output file %s\n", o_txt_filename);
		return 1;
	}
	
	////////////////////////////////////////////// DICTIONARY ////////////////////////////////////////////////////////

	unsigned int ints_per_row = ceil(num_cols/32.0);
	printf("Ints: %d\n", ints_per_row);

	// ALLOCATE MEMORY FOR OLD

	unsigned int *row = (unsigned int*) malloc(sizeof(unsigned int)*ints_per_row);
	
	if(row == NULL){
		printf("Error allocating memory for row.\n");
		return 1;
	}

	// DUPLICATE DICT

	int i = 0;
	int num_read;

	for (int j = 0; j<num_rows; j++) 
	{
		// READ OLD DICT

		for(i = 0; i <ints_per_row-1; i++) 
		{
			num_read = fscanf(i_f, "%u ", row+i);
			if(num_read != 1) 
			{
				printf("Error in scanf. %d %d\n", i, j);
				return 1;
			}
		}
		fscanf(i_f, "%u\n", row + i);
		
		// PRINT NEW DICT

		for(i = 0; i < ints_per_row*DUP - 1; i++)
		{
			fprintf(o_f, "%u ", *(row + i%ints_per_row));
		}	
		fprintf(o_f, "%u\n", *(row + i%ints_per_row));
		printProgress(j, num_rows, "Dictionary rows...");
	}

	// FREE

	free(row);
	fclose(i_f);
	fclose(o_f);

////////////////////////////////////////////// FILES /////////////////////////////////////////////////////////

	unsigned int num_cols_pad = ints_per_row*32;
	printf("Cols: %d\n", num_cols);

	// ALLOCATE MEMORY FOR OLD

	char **txt_rows = (char **) malloc(sizeof(char *)*num_cols);	
	
	if(txt_rows == NULL){
		printf("Error allocating memory for txt_row.\n");
		return 1;
	}	

	// DUPLICATE FILES

	// READ

	for(i=0; i<num_cols; i++)
	{
		printProgress(i, num_cols, "Reading files...");
		txt_rows[i] = (char *)malloc(sizeof(char)*MAX_FILE_NAME);
		fscanf(i_txt_f,"%[^\n]\n", txt_rows[i]);
	}

	// PRINT

	for(i = 0; i < num_cols_pad*DUP; i++)
	{
		printProgress(i, num_cols_pad*DUP, "Writing duplicated files...");
		if(i%num_cols_pad < num_cols)
			fprintf(o_txt_f, "%d_%s\n", i/num_cols_pad, txt_rows[i%num_cols_pad]);
		else
			fprintf(o_txt_f, "%d_%d_fakefile.txt\n", i/num_cols_pad, (i%num_cols_pad) - num_cols);
	}		

	// FREE

	for(i = 0; i<num_cols; i++) {
		free(txt_rows[i]);
	}
	free(txt_rows);
	fclose(i_txt_f);
	fclose(o_txt_f);

	return 0;
}

int main(int argc, char** argv) {

	// ARGUMENTS

	if(argc != ARG_COUNT+1){
		printf("%s\n", "Incorrect number of arguments passed.\nRequires <DUP> <input file> <output file> <input txt file> <output txt file> <num_cols> <num_rows>");
		return 1;
	}
	int DUP = 0;
	unsigned int num_cols = 0;
	unsigned int num_rows = 0;
	char *i_file = argv[2];
	char *o_file = argv[3];
	char *i_txt_file = argv[4];
	char *o_txt_file = argv[5];	
	int read_num = sscanf(argv[1], "%u", &DUP);
	if (read_num!=1)
		printf("%s\n", "Wrong number of arguments read from scanf.");
	read_num = sscanf(argv[6], "%u", &num_cols);
	if (read_num!=1)
		printf("%s\n", "Wrong number of arguments read from scanf.");
	read_num = sscanf(argv[7], "%u", &num_rows);
	if (read_num!=1)
		printf("%s\n", "Wrong number of arguments read from scanf.");

	// DUPLICATE

	if(DUP!=1)
	{
		printf("DUP = %d\n", DUP);
		return duplicate(DUP, i_file, o_file, i_txt_file, o_txt_file, num_cols, num_rows);
	}

	return 0;
}