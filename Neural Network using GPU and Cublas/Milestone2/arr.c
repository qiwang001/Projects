#define e 2.71828
#include "mnist.h"
#include<time.h>
#include "math.h"
#include <stdio.h>
#include"/opt/homebrew/Cellar/openblas/0.3.26/include/cblas.h"
#include<math.h>

int
main()
{
    int a[5][5];
    for(int i=0;i<5;i++)
        for(int j=0;j<5;j++)
            a[i][j]=5*i+j + 1;
    int * pa;
    pa=&a[1][0];
    printf("%d \n ", *a[0]);
    printf("%d ", a[0][1]);

}