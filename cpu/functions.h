#ifndef FUNCTIONS_H
#define FUNCTIONS_H



#define MATCH 3
#define MISMATCH -1
#define OPEN_GAP -5
#define EXTEND_GAP -1

typedef struct{
    int row;
    long long col;
    int maxScore;
}Result;





void initialize_sequence(char **seq, int len);

void get_memory_info();

int** initialize_matrix(int mtDNA_len, int nDNA_slice_len);

void fill_vector(int *vector, int value, int length);

void reverseString(char *str);

void local_alignment(int **H, int *E, int *F, char *reference, char *query, Result *result, int start_col);

void printMatrix(int **matrix, int rows, int cols);

char* read_from_file(const char *filename);

char* substring(const char* str, size_t start, size_t length);

void move_data(int **H, int mtDNA_len, int slice_len);

void copy_data(int **dst, int **src, int mtDNA_len);

#endif