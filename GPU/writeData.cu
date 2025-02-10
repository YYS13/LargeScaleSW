#include "functions.cuh"


int main(){
    // 指定 reference & query 大小
    size_t referenceLen = 3.2e8;
    size_t queryLen = 16569;

    char *reference = (char *)malloc(sizeof(char) * referenceLen);
    char *mtDNA = (char *)malloc(sizeof(char) * queryLen);

    // 初始化序列
    initialize_sequence(&reference,  (long long)(3.2e8));
    initialize_sequence(&mtDNA, 16569);

    //expand mtDNA because mtDNA is circular sequence
    char *query = expand_mtDNA(mtDNA);

    //write string to file
    save_to_file("../data/nDNA.txt", reference);
    save_to_file("../data/mtDNA.txt", query);

    free(reference);
    free(query);
    free(mtDNA);
    
    return 0;
}