#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv){
    for(int expand = 0; expand < 2; expand++){
        for(int i = 32; i <= 128; i += 32){
            char nDNA_Path[] = "../data/1.txt"; 
            char mtDNA_Path[] = "../data/mtDNA.txt"; 
            char command[100];
            snprintf(command, sizeof(command), "./gpu %s %s %d %d", mtDNA_Path, nDNA_Path, i, expand);
            system(command);
        }
    }

    system("python3 draw.py");
    return 0;
}