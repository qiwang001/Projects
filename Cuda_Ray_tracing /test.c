#include"stdlib.h"
#include"stdio.h"
#include <time.h>
#include <math.h>
#include "omp.h"
#define PI 3.1415926
int
main()
{
#define MAX(a,b) ((a)>(b)? (a):(b))
#define MIN(a,b) ((a)<(b)? (a):(b))
#define ABS(a) ((a)>0? (a):(-a))
#define PI 3.1415926
  unsigned int seed = omp_get_thread_num()*time(NULL);
  #pragma omp parallel for default(none) num_threads(2) 
  for(int i=0;i<5;i++)
    {
    unsigned int seed = time(NULL) ^ omp_get_thread_num();
    double phi = PI* (double)rand()/RAND_MAX;
    double cosine_theta = ((double)rand()/RAND_MAX - 0.5)*2;
    printf("threadID:%d\n, phi:%f\n, cosinetheta:%f\n",omp_get_thread_num(), phi, cosine_theta);
    }
    


}