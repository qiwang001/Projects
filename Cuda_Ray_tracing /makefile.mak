# Compile flags
CC = mpicc  nvcc
CFLAGS = -fopenmp -arch=sm_75 -lstdc++ -lcudart -L/programs/x86_64-linux/cuda/11.0/targets/x86_64-linux/lib/ -lm

# Object files
OBJS = main.o rfinal_sp_mpi.o
OBJS = main.o rfinal_dp_mpi.o

# Target executable
TARGET = program_sp
TARGET = program_dp
# Rule to compile a source file
.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to link the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)
