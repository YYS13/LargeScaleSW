#ifndef FUNCTIONS_H
#define FUNCTIONS_H



#define MATCH 3
#define MISMATCH -1
#define OPEN_GAP -5
#define EXTEND_GAP -1

typedef struct{
    int row;
    int col;
    int maxScore;
}Result;

typedef struct{
    int score;
    int direction;
}Cell;



void initialize_sequence(char **seq, int len);

Cell** initialize_matrix(char *reference, char *query);

void fill_vector(int *vector, int value, int length);

void reverseString(char *str);

void local_alignment(Cell **H, int *E, int *F, char *reference, char *query, Result *result);

void traceback(Cell **H, Result *result, char *reference, char *query);

void printMatrix(Cell **matrix, int rows, int cols);

#endif