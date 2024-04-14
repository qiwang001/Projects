#define e 2.71828
#include "mnist.h"
void shuffle(int arr[], int n) {
  srand(time(NULL)); // Seed the random number generator

  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1); // Generate random index within 0 to i
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }
}

int
main()
{
load_mnist();
  for(int j=0;j<784;j++){
    printf("%1.1f ", train_image[9999][j]);
    if(j%27==0)
    putchar('\n');}
}