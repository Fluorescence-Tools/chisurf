#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)
#define min(a,b) a <= b ? a : b

// Initialization function
void initValues (int nWorkItems, double *input, int nWorkGroups, double *output) {

  int i;

  for (i=0; i<nWorkItems; i++)
     input[i] = i+1;
       
  for (i=0; i<nWorkGroups; i++)
     output[i] = 0.0;
}

int main (int argc, char** argv) {

  // Size of vector 
  int size = atoi(argv[1]);;

  // Size of WorkGroup 
  int sizeWorkGroup = atoi(argv[2]);;

  // Arrays 
  double* xInput;
  double* sumReduction;

  // Final Sum  
  double finalSumSeq;
  double finalSumGPU;

  // Kernel file descriptor 
  FILE* fileKernel;

  // kernel code string
  char *kernelSource_str; 

  // Size of kernel source
  size_t source_size;

  // Index variable 
  int i;

  // Variables for clock 
  struct timeval chrono1, chrono2; 
  int micro_init, second_init;
  int micro_gpu_init, second_gpu_init; 
  int micro_gpu, second_gpu; 
  int micro_seq, second_seq; 

  // Total number of elements
  int nWorkItems = size;

  // Execute the OpenCL kernel on the list
  size_t global_item_size = size; 
  size_t local_item_size = sizeWorkGroup;
  
  // Number of work-groups
  int nWorkGroups = size/local_item_size;

  // Time start for preparing and initializing arrays
  gettimeofday(&chrono1, NULL);

  // Main values array
  xInput  = malloc(nWorkItems*sizeof(double));

  // Test allocation
  if (xInput == NULL)
    printf("Malloc failed for xInput array\n");

  // Allocate cumulative error array
  sumReduction = malloc(nWorkGroups*sizeof(double));

  // Test allocation
  if (sumReduction == NULL)
    printf("Malloc failed for sumReduction array\n");

  // Array initialization 
  initValues(nWorkItems, xInput, nWorkGroups, sumReduction);

  // Time end for preparing and initializing arrays
  gettimeofday(&chrono2, NULL);

  // Elapsed time for preparing and initializing arrays
  micro_init = chrono2.tv_usec - chrono1.tv_usec; 
  if (micro_init < 0) 
    { micro_init += 1000000; 
      second_init = chrono2.tv_sec - chrono1.tv_sec - 1;
    } 
    else 
      second_init = chrono2.tv_sec - chrono1.tv_sec;

  // Time start for preparing GPU/OpenCL
  gettimeofday(&chrono1, NULL);

  // Load the kernel source code into the array kernelSource_str
  fileKernel = fopen("sumReductionGPU.cl", "r");
  if (!fileKernel) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }

  // Read kernel code
  kernelSource_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(kernelSource_str, 1, MAX_SOURCE_SIZE, fileKernel);
  fclose(fileKernel);

  // OpenCL initialization 

  // Get platform and device information
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;   
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, 
      &device_id, &ret_num_devices);

  // Create an OpenCL context
  cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueueWithProperties( context, device_id, 0, &ret);

  // Create memory buffers on the device for each vector 
  cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
      nWorkItems * sizeof(double), NULL, &ret);	    
  if (ret != CL_SUCCESS)
    printf("Creating Input Buffer failed\n");

  cl_mem reductionBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
      nWorkGroups * sizeof(double), NULL, &ret);	    
  // Write to buffers
  ret = clEnqueueWriteBuffer(command_queue, inputBuffer, CL_TRUE, 0, 
        nWorkItems * sizeof(double), xInput, 0, NULL, NULL);	   	    
  ret = clEnqueueWriteBuffer(command_queue, reductionBuffer, CL_TRUE, 0, 
        nWorkGroups * sizeof(double), sumReduction, 0, NULL, NULL);	   	    

  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1, 
      (const char **)&kernelSource_str, (const size_t *)&source_size, &ret);

  // Build the program
  if (clBuildProgram(program, 1, &device_id, NULL, NULL, NULL) != CL_SUCCESS)
    { char buffer[1024];
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
      fprintf(stderr, "CL Compilation failed:\n%s", buffer);
      abort(); }

  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "sumGPU", &ret);

  // Set the arguments of the kernel
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);    
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&reductionBuffer);    
  clSetKernelArg(kernel, 2, local_item_size * sizeof(double),NULL);
     
  // Time end for preparing GPU/OpenCL
  gettimeofday(&chrono2, NULL);

  // Elapsed time for preparing GPU/OpenCL
  micro_gpu_init = chrono2.tv_usec - chrono1.tv_usec; 
  if (micro_gpu_init < 0) 
    { micro_gpu_init += 1000000; 
      second_gpu_init = chrono2.tv_sec - chrono1.tv_sec - 1;
    } 
    else 
      second_gpu_init = chrono2.tv_sec - chrono1.tv_sec;

  // GPU version
  // Time start for one GPU/NDRangeKernel call
  gettimeofday(&chrono1, NULL);

  // Execute kernel code
  if (clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
        &global_item_size, &local_item_size, 0, NULL, NULL) != CL_SUCCESS)
    fprintf(stderr,"Error in clEnqueueNDRangeKernel\n");

  // Read the buffer back to the array
  if (clEnqueueReadBuffer(command_queue, reductionBuffer, CL_TRUE, 0, 
        nWorkGroups * sizeof(double), sumReduction, 0, NULL, NULL) != CL_SUCCESS)
    fprintf(stderr,"Error in clEnqueueReadBuffer with reductionBuffer\n");

  // Final summation with CPU
  finalSumGPU = 0.0;
  for (i=0; i<nWorkGroups; i++)
     finalSumGPU += sumReduction[i];   

  // Time end for one GPU/NDRangeKernel call
  gettimeofday(&chrono2, NULL);

  // Elapsed time for one GPU/NDRangeKernel call
  micro_gpu = chrono2.tv_usec - chrono1.tv_usec; 
  if (micro_gpu < 0) 
    { micro_gpu += 1000000; 
      second_gpu = chrono2.tv_sec - chrono1.tv_sec - 1;
    } 
    else 
      second_gpu = chrono2.tv_sec - chrono1.tv_sec;

  // Sequential version
  // Time start for computing sum
  gettimeofday(&chrono1, NULL);

  finalSumSeq = 0.0;
  for (i=0; i<nWorkItems; i++)
     finalSumSeq += xInput[i];

  // Time end for computing sum
  gettimeofday(&chrono2, NULL);

  // Compute ellapsed time 
  micro_seq = chrono2.tv_usec - chrono1.tv_usec; 
  if (micro_seq < 0) 
    { micro_seq += 1000000; 
      second_seq = chrono2.tv_sec - chrono1.tv_sec - 1;
    } 
    else 
      second_seq = chrono2.tv_sec - chrono1.tv_sec;

  printf("\n");
  printf("  Problem size = %d\n", nWorkItems);
  printf("\n");
  printf("  Final Sum Sequential = %20.19e\n", finalSumSeq);
  printf("\n");
  printf("  Final Sum GPU = %20.19e\n", finalSumGPU);
  printf("\n");
  printf("  Initializing Arrays : Wall Clock = %d second %d micro\n", second_init, micro_init);
  printf("\n");
  printf("  Preparing GPU/OpenCL : Wall Clock = %d second %d micro\n", second_gpu_init, micro_gpu_init);
  printf("\n");
  printf("  Time for one NDRangeKernel call and WorkGroups final Sum  : Wall Clock = %d second %d micro\n", second_gpu, micro_gpu);
  printf("\n");
  printf("  Time for Sequential Sum computing  : Wall Clock = %d second %d micro\n", second_seq, micro_seq);
  printf("\n");

  // Free arrays
  free(xInput);
  free(sumReduction);

  // OpenCL Clean up
  clFinish(command_queue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(inputBuffer);    
  clReleaseMemObject(reductionBuffer);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return 0;
}
