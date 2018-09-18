CC = gcc -Wall
INC_GPU = -I/opt/AMDAPP/include
LD_GPU  = -lm -L/opt/AMDAPP/lib/x86_64/sdk -lOpenCL

all: sumReductionGPU 

sumReductionGPU : sumReductionGPU.c 
	$(CC) $(INC_GPU) $(LD_GPU) -o $@ sumReductionGPU.c

clean : 
	/bin/rm -f *.o sumReductionGPU 
