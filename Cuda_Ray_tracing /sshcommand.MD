ssh qiwang001@midway3.rcc.uchicago.edu
ouyuanshitiaohaogou~
scp rayTracing_final_ompSP.c qiwang001@midway3.rcc.uchicago.edu:~/ 
scp qiwang001@midway3.rcc.uchicago.edu:~/grid.txt ./
sinteractive --partition=gpu --gres=gpu:1 --account=mpcs51087 --time=00:5:00  --constraint=a100
nvcc rayTracing_finalSP.cu -o rsp -Xcompiler -fopenmp -lm

scp  qiwang001@midway3.rcc.uchicago.edu:~/rayTracing_final_omp.c ./
scp rayTracing_finalSP.cu qiwang001@midway3.rcc.uchicago.edu:~/
scp rayTracing_finalDP.cu qiwang001@midway3.rcc.uchicago.edu:~/
scp  qiwang001@midway3.rcc.uchicago.edu:~/rfinal_sp_mpi.cu ./
scp  qiwang001@midway3.rcc.uchicago.edu:~/rfinal_mpi.c ./
scp  qiwang001@midway3.rcc.uchicago.edu:~/rayTracing_finalSP.cu ./
nvcc -gencode arch=compute_70,code=compute_70 rayTracing_finalSP.cu -o rsp -Xcompiler -fopenmp -lm

scp rfinal_mpi.c qiwang001@midway3.rcc.uchicago.edu:~/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/programs/x86_64-linux/cuda/11.0/targets/x86_64-linux/lib
module load cuda openmpi
mpicc -c rfinal_mpi.c -o main.o
nvcc -c rfinal_sp_mpi.cu -arch=sm_75 -o rfinal_sp_mpi.o 

mpicc main.o rfinal_sp_mpi.o -lcudart -L/programs/x86_64-linux/cuda/11.0/targets/x86_64-linux/lib/ -lstdc++ -fopenmp -o program6000

mpirun -n 1 ./program 1000000000 16384 16384 256


module load cuda openmpi
mpicc -c rfinal_dp_mpi.c -o main.o
nvcc -c rfinal_dp_mpi.cu -arch=sm_70 -o rfinal_dp_mpi.o 

mpicc main.o rfinal_dp_mpi.o -lcudart -L/programs/x86_64-linux/cuda/11.0/targets/x86_64-linux/lib/ -lstdc++ -fopenmp -o programdp

mpirun -n 1 ./programdp 1000000000 16384 16384 256