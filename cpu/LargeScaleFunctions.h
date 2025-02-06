#ifndef LARGESCALEFUNCTIONS_H
#define LARGESCALEFUNCTIONS

#include <stddef.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define MATCH 3
#define MISMATCH -1
#define OPEN_GAP -5
#define EXTEND_GAP -1

typedef struct{
    int row;
    int col;
    int maxScore;
}Result;


void initialize_sequence(char **seq, long long len);

char* expand_mtDNA(char *mtDNA);

void get_memory_info();

int* initialize_vector(long long len, int val); 

void local_alignment(int *H, int *E, char *reference, char *query, Result *result);

void convert_time(double total_seconds);

char* read_from_file(const char *filename);

char* substring(const char* str, size_t start, size_t length);








#endif