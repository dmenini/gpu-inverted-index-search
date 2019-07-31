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
#define INV_DICT_REMAINDER	((unsigned int)(N_FILES % (sizeof(int)*8)))
#define INV_DICT_SIZE		(D*INV_DICT_WIDTH)
#define BLOCK_SIZE 			256
#define N_BLOCKS			ceil(N_FILES/(sizeof(int)*8*BLOCK_SIZE))
#define N_THREADS 			(BLOCK_SIZE*N_BLOCKS) 

unsigned int *inv_dictionary;
char files[N_FILES][MAX_FILE_NAME];
char query[D];
unsigned int matches[N_FILES];

// inv_dict_width = width of inv dictionary in number of elements
// int_size = size of an integer in bytes
__global__ void queryKernel(unsigned int * __restrict__ inv_dictionary, unsigned int inv_dict_width, char * __restrict__ query, unsigned int * matches, unsigned int int_size, unsigned int n_threads) {
	unsigned short curr_bit = 0;
	unsigned int match_pos;
	unsigned int match_cnt = 0;
	unsigned int match_idx;
	unsigned int thread_matches = 0;

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	// for (int i = 0; i<D; i++) {
	// 	for(int j = 0 ; j<inv_dict_width; j++) {
	// 		unsigned int err = *(inv_dictionary + i*inv_dict_width + j);
	// 	}
	// }

	while(index < inv_dict_width){ //n_threads = 512, inv_dict_width = 657
		match_idx = 0xFFFFFFFF;
		for(int i = 0; i < D; i++) {
			if (query[i] == '1') {
				// printf("Query: %d\n", i);
				// printf("match[%u]: %u\tdict[%u]: %u\t", j, match_idx[j], j, inv_dictionary[i][j]);
				// printf("Index:\t%d\nMatch index:\t%u\nInv dict:\t%d\n", index, match_idx, inv_dictionary[i][index]);
				match_idx = match_idx & *(inv_dictionary + i*inv_dict_width + index);
				// printf("Match index:\t%d\n", match_idx);
				// printf("res[%u]: %u\n", j, match_idx[j]);
			}
		}
		match_cnt = 0;
		for(int j=1; j<=(int_size*8) || match_idx>0; j++) {
			curr_bit = match_idx%2;
			match_idx = match_idx/2;
			if(curr_bit==1) {
				// printf("%u", i);
				match_pos = (index + 1)*int_size*8 - j + 1; // starts indexing at 1 not 0
				matches[index*int_size*8+match_cnt] = match_pos;
				// printf("(%u) ", file_index);
				// strcpy(matches[match_cnt], files[file_index]);
				match_cnt++;
			}
		}
		index += n_threads;
		thread_matches++;
	}
	// printf("ended");
}


__host__ void load_inv_dictionary(char file[]){

	unsigned int num_read;
	FILE *fp = NULL;
	fp = fopen(file, "r");

	if (fp == NULL){
		printf("Error while opening %s\n", file);
	}

	// int size = fsize(fp);
	// printf("Size: %d\n", size);
	// printf("INV_DICT_SIZE: %d\n", INV_DICT_SIZE);

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
		printf("Requires arguments: <dictionary input file> <query input file> <files file> <output directory>\n");
		return 1;
	}	

	printf("Dict file:\t%s\n", argv[1]);
	printf("Query file:\t%s\n", argv[2]);
	printf("Files file:\t%s\n", argv[3]);	
	printf("Out dir:\t%s\n", argv[4]);	

	char dict_file[MAX_FILE_NAME];  
	char files_file[MAX_FILE_NAME];  
	char query_file[MAX_FILE_NAME];
	char output_dir[MAX_FILE_NAME];
	char report_file[MAX_FILE_NAME];

	sprintf(dict_file, "%s", argv[1]);
	sprintf(query_file, "%s", argv[2]);
	sprintf(files_file, "%s", argv[3]);
	sprintf(output_dir, "%s", argv[4]);

	sprintf(report_file, "%s/cu_query_report.txt", output_dir);

	printf("Rep file:\t%s\n", report_file);

	unsigned int *d_inv_dict, *d_matches;
	char *d_query;

	printf("INV_DICT_SIZE:\t%d\nINV_DICT_WIDTH:\t%d\nINV_DICT_REMAINDER:\t%d\n", INV_DICT_SIZE, INV_DICT_WIDTH, INV_DICT_REMAINDER);
	

	inv_dictionary = (unsigned int *)malloc(INV_DICT_SIZE*sizeof(unsigned int));
	// inv_dictionary = (unsigned int **)malloc(D*sizeof(unsigned int *));
	// for (int i=0; i<D; i++) {
	// 	inv_dictionary[i] = (unsigned int *)malloc(INV_DICT_WIDTH*sizeof(unsigned int));
	// }

	load_inv_dictionary(dict_file);
	load_query(query_file);

	// printf("Printing query...\n");
	// for (int i =0; i<D; i++) {
	// 	if(query[i] == '1')
	// 		printf("i:\t%d\n", i);
	// }

	cudaMalloc((void **)&d_inv_dict, INV_DICT_SIZE*sizeof(unsigned int));
	cudaMalloc((void **)&d_query, D);
	cudaMalloc((void **)&d_matches, N_FILES*sizeof(unsigned int));

	printf("Allocated arrays...\n");

	// for (int j =0; j<100; j++) {
	// 	for (int i =0; i<INV_DICT_WIDTH; i++) {
	// 		printf("%d:\t%d:\t%u:\t\n", j, i, *(inv_dictionary + j*INV_DICT_WIDTH + i));
	// 	}
	// }

	cudaMemcpy(d_query, query, D*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemset(d_matches, 0, N_FILES*sizeof(unsigned int));
	cudaError_t err = cudaMemcpy(d_inv_dict, inv_dictionary, INV_DICT_SIZE*sizeof(unsigned int), cudaMemcpyHostToDevice);

	printf("Err: %d\n", err);
	printf("Query...\n");

	queryKernel<<<N_BLOCKS, BLOCK_SIZE>>>((unsigned int *)d_inv_dict, INV_DICT_WIDTH, d_query, d_matches, sizeof(int), N_THREADS);

	cudaMemcpy(matches, d_matches, N_FILES*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	printMatches();

	load_files(files_file);
	reportQuery(report_file);

	free(inv_dictionary);
	cudaFree(d_inv_dict);
	cudaFree(d_query);
	cudaFree(d_matches);

	return 0;
}