#include "LargeScaleFunctions.c"


int main(){
    clock_t start, end;
    start = clock();

    get_memory_info();
    char *reference, *mtDNA, *query;
    srand((unsigned int)time(NULL));
    //initialize sequence
    // reference = "ACGTAGCATACATTAGTATTTTCATCGTACTGCATATCGATGTATGCATGTATTT";
    // query = "GTATGCATCGATCGATCACGATCTACGGCTAGC";
    initialize_sequence(&reference,  (long long)(3.2e8));
    initialize_sequence(&mtDNA, 16569);

    //expand mtDNA because mtDNA is circular sequence
    query = expand_mtDNA(mtDNA);
    

    //initialize vector
    int *H = initialize_vector(strlen(reference) + 1, 0);
    int *E = initialize_vector(strlen(reference) + 1, INT_MIN);


    Result result;
    result.maxScore = INT_MIN;

    //alignment
    local_alignment(H, E, reference, query, &result);
    printf("maxScore = %d at (%d, %d)\n", result.maxScore, result.row, result.col);

    // printf("reference : %s\n", reference);
    // printf("query : %s\n", query);



    //free memory space
    free(reference);
    free(query);
    free(H);
    free(E);

    end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;  // 计算耗时（秒）
    printf("total time: %.6f seconds\n", elapsed_time);
    
    convert_time(elapsed_time);


    return 0;
}