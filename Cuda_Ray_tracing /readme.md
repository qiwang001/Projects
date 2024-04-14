For Mpi program,use "-L/programs/x86_64-linux/cuda/11.0/targets/x86_64-linux/lib/" to link the two .o files.

Before running the mpi program, export below varible to environmental variable:
"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/programs/x86_64-linux/cuda/11.0/targets/x86_64-linux/lib"
