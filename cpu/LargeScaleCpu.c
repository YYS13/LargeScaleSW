#include "LargeScaleFunctions.c"


int main(){
    clock_t start, end;
    start = clock();

    get_memory_info();
    char *nDNA, *mtDNA;
    srand((unsigned int)time(NULL));
    //initialize sequence
    // reference = "ACGTAGCATACATTAGTATTTTCATCGTACTGCATATCGATGTATGCATGTATTT";
    // query = "GTATGCATCGATCGATCACGATCTACGGCTAGC";
    // initialize_sequence(&reference,  (long long)(3.2e8));
    // initialize_sequence(&mtDNA, 16569);
    nDNA = read_from_file("../data/nDNA.txt");
    mtDNA = read_from_file("../data/mtDNA.txt");
    //test data
    char *nDNA_slice = substring(nDNA, 0, 78);
    char *mtDNA_slice = substring(mtDNA, 0, 43);
    // printf("nDNA : %s\n", nDNA_slice);
    // printf("mtDNA : %s\n", mtDNA_slice);
    

    //expand mtDNA because mtDNA is circular sequence
    // query = expand_mtDNA(mtDNA);
    

    //initialize vector
    int *H = initialize_vector(strlen(nDNA) + 1, 0);
    int *E = initialize_vector(strlen(nDNA) + 1, INT_MIN - EXTEND_GAP);


    Result result;
    result.maxScore = INT_MIN;

    //alignment
    local_alignment(H, E, nDNA_slice, mtDNA_slice, &result);
    printf("maxScore = %d at (%d, %d)\n", result.maxScore, result.row, result.col);



    //free memory space
    free(nDNA);
    free(mtDNA);
    free(nDNA_slice);
    free(mtDNA_slice);
    free(H);
    free(E);

    end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;  // 计算耗时（秒）
    printf("total time: %.6f seconds\n", elapsed_time);
    
    convert_time(elapsed_time);


    return 0;
}