#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]){
    for(int expand = 0; expand < 2; expand++){
        char nDNA_Path[20]; 
        snprintf(nDNA_Path, sizeof(nDNA_Path), "../data/%s", argv[1]);
        char mtDNA_Path[] = "../data/mtDNA.txt";
        for(int i = 32; i <= 128; i += 32){ 
            char command[100];
            snprintf(command, sizeof(command), "./gpu %s %s %d %d", mtDNA_Path, nDNA_Path, i, expand);
            system(command);
        }
    }

    system("python3 draw.py");
    return 0;
}