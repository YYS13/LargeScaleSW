#include <time.h>
#include <math.h>

#include "functions.c"


int main(){
    // load sequence
    char *nDNA = read_from_file("../data/nDNA.txt");
    char *mtDNA = read_from_file("../data/mtDNA.txt");
    printf("nDNA length = %zu\n", strlen(nDNA));
    printf("mtDNA length = %zu\n", strlen(mtDNA));

    nDNA = substring(nDNA, 0, 7890);
    //mtDNA = substring(mtDNA, 0, 10);

    //決定片段長度
    int nDNA_slice_len = 2000;

    //initialize matrix
    int **H = initialize_matrix(strlen(mtDNA), nDNA_slice_len);

    //initialize E & F
    int *E = malloc(sizeof(int) * nDNA_slice_len + 1);
    int *F = malloc(sizeof(int) * (strlen(mtDNA) + 1));

    //fill element with negative infinity
    fill_vector(F, INT_MIN - EXTEND_GAP, (strlen(mtDNA) + 1));
    

    Result result;
    result.maxScore = INT_MIN;

    //開始計算
    for(int epoch = 0; epoch <= strlen(nDNA) - nDNA_slice_len; epoch += nDNA_slice_len){
        printf("Epoch %d/%d\n", epoch/nDNA_slice_len + 1, (int)(strlen(nDNA)/nDNA_slice_len));
        // 取 substring
        char *slice = substring(nDNA, epoch, nDNA_slice_len);
        // fill E with -∞
        fill_vector(E, INT_MIN - EXTEND_GAP, (strlen(slice) + 1));
        // caculate submatrix
        local_alignment(H, E, F, slice, mtDNA, &result, epoch * nDNA_slice_len);
        //printf("H: \n");
        //printMatrix(H, strlen(mtDNA) + 1, nDNA_slice_len + 1);
        //move data to first row
        move_data(H, strlen(mtDNA), nDNA_slice_len);
        free(slice);
    }
    

    //計算剩下沒有被整除的部份
    if(strlen(nDNA) % nDNA_slice_len != 0){
        int rest_len = strlen(nDNA) % nDNA_slice_len;
        char *rest_slice = substring(nDNA, strlen(nDNA) - rest_len, rest_len);
        int *rest_E = malloc(sizeof(int) * (rest_len + 1));
        fill_vector(rest_E, INT_MIN - EXTEND_GAP, (rest_len + 1));
        int **rest_H = initialize_matrix(strlen(mtDNA), rest_len);
        copy_data(rest_H, H, strlen(mtDNA));
        local_alignment(rest_H, rest_E, F, rest_slice, mtDNA, &result, (strlen(nDNA) / nDNA_slice_len) * nDNA_slice_len);
        //printf("rest_H: \n");
        //printMatrix(rest_H, strlen(mtDNA) + 1, rest_len + 1);

        free(rest_slice);
        free(rest_E);
        free(rest_H);
    }

    printf("max score = %d at (%d, %lld)\n", result.maxScore, result.row, result.col);

    // printf("\nE: \n");
    // printMatrix(E, strlen(query) + 1, strlen(reference) + 1);
    // printf("\nF: \n");
    // printMatrix(F, strlen(query) + 1, strlen(reference) + 1);



    //free memory space
    free(nDNA);
    free(mtDNA);
    for (int i = 0; i <= strlen(mtDNA); i++) {
        free(H[i]);
    }
    free(H);
    free(E);
    free(F);

    return 0;
}