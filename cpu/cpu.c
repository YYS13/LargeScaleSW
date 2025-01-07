#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "functions.c"


int main(){
    char *reference, *query;
    srand((unsigned int)time(NULL));
    //initialize sequence
    initialize_sequence(&reference, 30);
    initialize_sequence(&query, 12);

    

    //initialize matrix
    Cell **H = initialize_matrix(reference, query);

    //initialize E & F
    int *E = malloc(sizeof(int) * (strlen(reference) + 1));
    int *F = malloc(sizeof(int) * (strlen(query) + 1));

    //fill element with negative infinity
    fill_vector(E, INT_MIN - EXTEND_GAP, (strlen(reference) + 1));
    fill_vector(F, INT_MIN - EXTEND_GAP, (strlen(query) + 1));
    printf("%d\n", E[1]);

    Result result;
    result.maxScore = INT_MIN;

    //alignment
    local_alignment(H, E, F, reference, query, &result);

    printf("reference : %s\n", reference);
    printf("query : %s\n", query);
    printf("H: \n");
    printMatrix(H, strlen(query) + 1, strlen(reference) + 1);
    // printf("\nE: \n");
    // printMatrix(E, strlen(query) + 1, strlen(reference) + 1);
    // printf("\nF: \n");
    // printMatrix(F, strlen(query) + 1, strlen(reference) + 1);


    //traceback
    traceback(H, &result, reference, query);

    //free memory space
    free(reference);
    free(query);
    for (int i = 0; i <= strlen(query); i++) {
        free(H[i]);
    }
    free(H);
    free(E);
    free(F);

    return 0;
}