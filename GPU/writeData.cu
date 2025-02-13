#include "functions.cuh"


int main(){
    // 讀取數據
    char *nDNA = read_from_file("../data/GRCh38.chromosome2.txt");
    char *nc1 = read_from_file("../data/NC_012920.1.txt");
    printf("nDNA length = %zu\n", strlen(nDNA));
    printf("NC_012920.1 length = %zu \n", strlen(nc1));

    //檢查 nDNA 中有無 N 並隨機替換成 ACGT
    replaceN(nDNA);

    //expand mtDNA because mtDNA is circular sequence
    char *mtDNA = expand_mtDNA(nc1);
    printf("mtDNA length = %zu\n",strlen(mtDNA));

    //write string to file
    save_to_file("../data/nDNA.txt", nDNA);
    save_to_file("../data/mtDNA.txt", mtDNA);

    free(nDNA);
    free(nc1);
    free(mtDNA);
    
    return 0;
}