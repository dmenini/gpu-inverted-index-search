#include <stdio.h>
#include <string.h>
#include <math.h>

#ifndef D
	#define D 			10000
#endif

#ifndef N_FILES
	#define N_FILES 	21000
#endif

#define ARG_COUNT 			4
#define MAX_FILE_NAME 		100

#define INV_DICT_WIDTH		((unsigned int)(ceil(N_FILES / (float)(sizeof(int)*8))))
// #define INV_DICT_REMAINDER	((unsigned int)(N_FILES % (sizeof(int)*8)))
#define INV_DICT_SIZE		(D*INV_DICT_WIDTH)
#define BLOCK_SIZE 			64
#define N_BLOCKS			((unsigned int)(ceil(N_FILES/ (float)(sizeof(int)*8*BLOCK_SIZE))))
#define N_THREADS 			(BLOCK_SIZE*N_BLOCKS) 

unsigned int *inv_dictionary;
char files[N_FILES][MAX_FILE_NAME];
char query[D];
unsigned int query_ones[D];
unsigned int matches[N_FILES];

// inv_dict_width = width of inv dictionary in number of elements
// int_size = size of an integer in bytes
__global__ void queryKernel(unsigned int * __restrict__ inv_dictionary, unsigned int inv_dict_width, const unsigned int * __restrict__ query_ones, unsigned int ones_cnt, unsigned int * matches, unsigned int int_size, unsigned int n_threads) {
	unsigned short curr_bit = 0;
	unsigned int match_pos;
	unsigned int match_cnt = 0;
	unsigned int match_idx;

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

	// while(index < inv_dict_width){ //n_threads = 512, inv_dict_width = 657
	if (index < inv_dict_width) {
		match_idx = 0xFFFFFFFF;
		for(int i = 0; i < ones_cnt; i++) {			
			// if (query[i] == '1') {
				match_idx = match_idx & *(inv_dictionary + query_ones[i]*inv_dict_width + index);
				// if(match_idx == 0)
					// break;
			// }
		}

		match_cnt = 0;
		for(int j=1; j<=(int_size*8) && match_idx>0; j++) {
			curr_bit = match_idx%2;
			match_idx = match_idx/2;
			if(curr_bit==1) {
				match_pos = (index + 1)*int_size*8 - j + 1; // starts indexing at 1 not 0
				matches[index*int_size*8+match_cnt] = match_pos;
				match_cnt++;
			}
		}
		// index += n_threads;
	}
}


__host__ void load_inv_dictionary(char file[]){

	unsigned int num_read;
	FILE *fp = NULL;
	fp = fopen(file, "r");

	if (fp == NULL){
		printf("Error while opening %s\n", file);
	}

	int i;
	for(int j = 0; j<D; j++) {
		for(i=0; i<INV_DICT_WIDTH-1; i++){
			// printf("%d:\t%d:\t", j, i);
			num_read = fscanf(fp, "%u ", inv_dictionary + j*INV_DICT_WIDTH + i);
			// printf("%u\n", *(inv_dictionary + j*INV_DICT_WIDTH + i));

			if (num_read != 1) 
				printf("ERROR!\t fscanf did not fill all arguments\n");
		}
		// printf("%d:\t%d:\t", j, i);
		num_read = fscanf(fp, "%u\n", inv_dictionary + j*INV_DICT_WIDTH + i);
		// printf("%u\n", *(inv_dictionary + j*INV_DICT_WIDTH + i));
		if (num_read != 1) 
			printf("ERROR!\t fscanf did not fill all arguments\n");
	}

	fclose(fp);
}

__host__ void load_query(char file[]){

	unsigned int num_read;
	FILE *fp = NULL;
	fp = fopen(file, "r");

	if (fp == NULL){
		printf("Error while opening %s\n", file);
	}

	for (int i=0; i<D; i++) {
		num_read = fscanf(fp, "%c", &query[i]);
		if (num_read != 1) 
			printf("ERROR!\t fscanf did not fill all arguments\n");
	}

	// printf("Loaded query: ");

	// for (int i=0; i<D; i++) {
	// 	printf("%c", query[i]);
	// }

	// printf("\n");

	fclose(fp);
}

__host__ void printMatches(){
	for (int i=0; i<N_FILES; i++) {
		printf("Match %d:\t%u\n", i, matches[i]);
	}
}

__host__ void load_files(char file[]) {
	FILE *fp = NULL;
	fp = fopen(file, "r");
	int num_read = 0;

	if (fp == NULL){
		printf("Error while opening %s\n", file);
	}

	for(int i = 0; i<N_FILES; i++) {
		num_read = fscanf(fp, "%s", files[i]);
		if (num_read != 1)
			printf("Error reading file: %s\n", file);
		// printf("%d: %s\n", i, files[i]);
	}
	fclose(fp);
}

unsigned int findQueryOnes(char* query){
	unsigned int ones_cnt = 0;
	for(unsigned int i=0; i<D; i++) {
		if(query[i] == '1')
		{
			query_ones[ones_cnt] = i;
			// printf("One position:\t%d\n", i);
			ones_cnt++; 
		}
	}
	return ones_cnt;
}


__host__ void reportQuery(char* report_f){
	FILE *rep = fopen(report_f, "w+");

	for (int i=0; i < N_FILES; i++){
		if (matches[i] != 0)
		{
			fprintf(rep, "%s\n", files[matches[i]-1]);
		}
	}
	fclose(rep);
}


int main(int argc, char** argv){

	// Command-line arguments:
	// D, itemmem input file, dictionary input file, query
	if(argc != (ARG_COUNT+1)){
		printf("Requires arguments: <files file> <dictionary input file> <query input file> <output directory>\n");
		return 1;
	}	

	char dict_file[MAX_FILE_NAME];  
	char files_file[MAX_FILE_NAME];  
	char query_file[MAX_FILE_NAME];
	char output_dir[MAX_FILE_NAME];
	char report_file[MAX_FILE_NAME];

	sprintf(files_file, "%s", argv[1]);
	sprintf(dict_file, "%s", argv[2]);
	sprintf(query_file, "%s", argv[3]);
	sprintf(output_dir, "%s", argv[4]);

	printf("INV_DICT_SIZE:\t%u\nINV_DICT_WIDTH:\t%u\nN_BLOCKS:\t%u\nBLOCK_SIZE:\t%d\n", INV_DICT_SIZE, INV_DICT_WIDTH, N_BLOCKS, BLOCK_SIZE);

	sprintf(report_file, "%s/cu_query_report.txt", output_dir);

	unsigned int *d_inv_dict, *d_matches;
	unsigned int *d_query_ones;
	unsigned int ones_cnt;

	inv_dictionary = (unsigned int *)malloc(INV_DICT_SIZE*sizeof(unsigned int));

	load_inv_dictionary(dict_file);
	load_query(query_file);
	ones_cnt = findQueryOnes(query);


#ifdef PROFILING
	for (int i = 0; i < 100; ++i)
	{
#endif	
		cudaMalloc((void **)&d_inv_dict, INV_DICT_SIZE*sizeof(unsigned int));
		cudaMalloc((void **)&d_query_ones, ones_cnt*sizeof(unsigned int));
		cudaMalloc((void **)&d_matches, N_FILES*sizeof(unsigned int));

		// printf("Allocated arrays...\n");

		cudaMemcpy(d_query_ones, query_ones, ones_cnt*sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemset(d_matches, 0, N_FILES*sizeof(unsigned int));
		cudaError_t err = cudaMemcpy(d_inv_dict, inv_dictionary, INV_DICT_SIZE*sizeof(unsigned int), cudaMemcpyHostToDevice);

		// printf("Err: %d\n", err);
		// printf("Query...\n");

		queryKernel<<<N_BLOCKS, BLOCK_SIZE>>>((unsigned int *)d_inv_dict, INV_DICT_WIDTH, d_query_ones, ones_cnt, d_matches, sizeof(int), N_THREADS);

		cudaMemcpy(matches, d_matches, N_FILES*sizeof(unsigned int), cudaMemcpyDeviceToHost);

		// printMatches();

		load_files(files_file);
		reportQuery(report_file);

		cudaFree(d_inv_dict);
		cudaFree(d_query_ones);
		cudaFree(d_matches);
#ifdef PROFILING
	}
#endif
	free(inv_dictionary);


	printf("Ended! :)\n");

	return 0;
}