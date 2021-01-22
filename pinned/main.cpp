#include <fstream>
#include <iostream>
#include <exception>
#include <iomanip>
#include <array>
#include <stdio.h>
#include <vector>

#include <stdlib.h>
#include <time.h>
#include <string>


#define CL_HPP_TARGET_OPENCL_VERSION 110

#define __CL_ENABLE_EXCEPTIONS
#define __MAX_DEFAULT_VECTOR_SIZE 128

#if defined(__APPLE__) || defined(MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/opencl.h>  
   #include <CL/cl.hpp>   
#endif

const int N = 100000;
const int size = 100000;


////////////////
std::string tmpstr;
cl_uint tmp;
cl_bool bool_tmp;
cl_device_fp_config dfpc_tmp;

cl_uint ocl_plat_idx = 0; // opencl platform index
cl_uint ocl_dev_idx = 0; // opencl device index

cl_uint clui_num_platforms = 0;
cl_uint clui_num_devices = 0;

cl_int cli_err_num;

cl_device_type device_type;
//////////////////////


const char *getErrorString(cl_int error);
int GetPlatformInfo(std::vector<cl::Platform> platforms, std::vector<cl::Device> devices);
void printArray(std::string header, const int *vec);
void fillArray(float *vec, int _seed);
void CreateMatrix(std::array<std::array<int, N>, N>& matrix, int num);
void PrintStdArray(const std::array<std::array<int, N>, N> &array);

int main(int argc, char* argv[]){
    size_t datasize = N * sizeof(float);


    // A + B creationg
    std::array<float, N> A;
    std::array<float, N> B;
    std::array<float, N> C;
    

    cl::Program program;
    std::vector<cl::Device> devices;
  

    try{
        std::vector<cl::Platform> platforms; // to musi byc tutaj!!!
        cl::Platform::get(&platforms);
//////////////
        GetPlatformInfo(platforms, devices);

        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

            
        cl::Context context(devices);

        cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);    

        std::string pwd_path = getenv("PWD");
        std::string file_rel_path = "/../sum.cl";
        std::string path = pwd_path+file_rel_path;
    

        std::ifstream sourceFile(path);
        std::string sourceCode(std::istreambuf_iterator <char>(sourceFile),(std::istreambuf_iterator <char>()));

        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
       
        program = cl::Program(context, source);

        program.build(devices);

        cl::Kernel kernel_vector_sum(program, "vector_sum");
        cl::Kernel kernel_vector_modulo(program, "vector_modulo");

        //////


        cl::Buffer bufferA(context,CL_MEM_READ_WRITE, N * sizeof(float));
        cl::Buffer bufferB(context,CL_MEM_READ_WRITE, N * sizeof(float));
        cl::Buffer bufferC(context,CL_MEM_READ_WRITE, N * sizeof(float));

        


        kernel_vector_sum.setArg(0, bufferA);
        kernel_vector_sum.setArg(1, bufferB);
        kernel_vector_sum.setArg(2, bufferC);
      

        kernel_vector_modulo.setArg(0, bufferA);
        kernel_vector_modulo.setArg(1, bufferB);


        float * hA = static_cast<float*>(queue.enqueueMapBuffer(bufferA, CL_TRUE, CL_MAP_WRITE, 0, N * sizeof(float)));
        float * hB = static_cast<float*>(queue.enqueueMapBuffer(bufferB, CL_TRUE, CL_MAP_WRITE, 0, N * sizeof(float)));
        float * hC = static_cast<float*>(queue.enqueueMapBuffer(bufferC, CL_TRUE, CL_MAP_WRITE, 0, N * sizeof(float)));
        
        
        for(int i = 0; i < N; i++){
            hA[i] = 0.3 * i;
            hB[i] = i*0.2;
            //std::cout << "A[" << i << "]=" << hA[i] << " B[" << i << "]=" << hB[i] << std::endl;
        }
        queue.enqueueUnmapMemObject(bufferA, hA);
        queue.enqueueUnmapMemObject(bufferB, hB);
        queue.enqueueUnmapMemObject(bufferC, hC);
        


        int max = 500000;

        for(int i = 0; i < max; i++){
            queue.enqueueNDRangeKernel(kernel_vector_sum, cl::NullRange, cl::NDRange(N));
            // read from buffer slow method
            //queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, N * sizeof(float), C.data());

            // read from buffer fast method
            //float * out = static_cast<float*>(queue.enqueueMapBuffer(bufferC, CL_TRUE, CL_MAP_READ, 0, N * sizeof(float)));
            
            //for(int j = 0; j < N; j++){
                //std::cout << "out[" << j << "]=" << out[j] << std::endl;
            //}
            //queue.enqueueUnmapMemObject(bufferC, out);
            
            queue.enqueueNDRangeKernel(kernel_vector_modulo, cl::NullRange, cl::NDRange(N));

            //for(int j = 0; j < N; j++){
                //std::cout << "A[" << j << "]=" << A[j] << " B[" << j << "]=" << B[j] << std::endl;
            //}
        }
        

       
    }catch(cl::Error e){
         std::cout << e.what() << ": Error code " << e.err() <<  getErrorString(e.err()) << std::endl;

         if (e.err() == CL_BUILD_PROGRAM_FAILURE){
            for (cl::Device device : devices){
                
                cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
                
                if (status != CL_BUILD_ERROR)
                    continue;

                std::string name   = device.getInfo<CL_DEVICE_NAME>();
                std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << "Build log for " << name << ": " << std::endl << buildlog << std::endl;
            }
        }
        else{
            std::cout << " test 7" << std::endl;
            throw e;
        }

    }
    
    return 0;
}
/*
int main(int argc, char** argv)
{
	cl_int error;

	// Get the OpenCL platform and device
	cl_platform_id platform;
	cl_device_id device;
	cl_uint num_platforms;
	cl_uint num_devices;

	error = clGetPlatformIDs(1, &platform, &num_platforms);
	if (error)
	{
		printf("Error getting platform ID: %d\n", error);
		return error;
	}

	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
	if (error)
	{
		printf("Error getting device ID: %d\n", error);
		return error;
	}

	char* device_name = get_device_info(device, CL_DEVICE_NAME);

	if (device_name)
		printf("Running on %s", device_name);
	else
		printf("Error getting device name, but continuing\n");

	// Create a context, a command queue
	cl_context context;
	cl_command_queue queue;

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
	if (error)
	{
		printf("Error creating context: %d", error);
		return error;
	}

	queue = clCreateCommandQueue(context, device, NULL, &error);
	if (error)
	{
		printf("Error creating the command queue: %d", error);
		return error;
	}

	// Build the OpenCL program and create our kernels
	cl_program program;
	cl_kernel kernel_vector_sum;
	cl_kernel kernel_vector_modulo;

	// Create the program
	const char* cl_source = kernel;
	size_t cl_length = strlen(kernel);

	program = clCreateProgramWithSource(context, 1, &cl_source, &cl_length, &error);
	if (error)
	{
		printf("Failed to create OpenCL program: %d", error);
		return error;
	}

	// Compiled under OpenCL 2.0, to precisely reproduce the issue
	error = clBuildProgram(program, 1, &device, "-cl-std=CL2.0", NULL, NULL);
	if (error)
	{
		// If there's a compilation error, print out the message
		cl_int build_err = error;
		size_t log_size;
		char* build_log;

		error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		build_log = (char*)malloc(log_size);
		error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);

		printf("\n%s", build_log);
		return build_err;
	}

	// Finally, create the kernel
	kernel_vector_sum = clCreateKernel(program, "vector_sum", &error);
	kernel_vector_modulo = clCreateKernel(program, "vector_modulo", &error);
	if (error)
	{
		printf("Failed to create the kernel: %d", error);
		return error;
	}

	// Create required memory buffers and initialize the inputs
	cl_mem a_mem; // Input
	cl_mem b_mem; // Input
	cl_mem c_mem; // Output

	a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_ELEMENTS * sizeof(cl_float), NULL, &error);
	b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_ELEMENTS * sizeof(cl_float), NULL, &error);
	c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NUM_ELEMENTS * sizeof(cl_float), NULL, &error);

	float* a_buffer = (float*)malloc(NUM_ELEMENTS * sizeof(cl_float));
	float* b_buffer = (float*)malloc(NUM_ELEMENTS * sizeof(cl_float));

	srand(time(0));

	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		a_buffer[i] = (float)rand() / RAND_MAX;
		b_buffer[i] = (float)rand() / RAND_MAX;
	}

	// Copy the inputs into the device buffers and set kernel arguments
	error  = clEnqueueWriteBuffer(queue, a_mem, CL_TRUE, 0, NUM_ELEMENTS * sizeof(cl_float), a_buffer, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(queue, b_mem, CL_TRUE, 0, NUM_ELEMENTS * sizeof(cl_float), b_buffer, 0, NULL, NULL);
	if (error)
	{
		printf("Failed to write inputs to device buffers: %d", error);
		return error;
	}

	error  = clSetKernelArg(kernel_vector_sum, 0, sizeof(cl_mem), &a_mem);
	error |= clSetKernelArg(kernel_vector_sum, 1, sizeof(cl_mem), &b_mem);
	error |= clSetKernelArg(kernel_vector_sum, 2, sizeof(cl_mem), &c_mem);
	if (error)
	{
		printf("Failed to set kernel arguments: %d", error);
		return error;
	}

	// Set up the worksize
	size_t global_work_offset = 0;
	size_t global_work_size = NUM_ELEMENTS;
	size_t local_work_size = 256;

	cl_mem tmp_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, COPY_SIZE * sizeof(cl_float), NULL, &error);
	void* test_array = clEnqueueMapBuffer(queue, tmp_buf, CL_TRUE, CL_MAP_READ,
		0, COPY_SIZE * sizeof(cl_float), 0, NULL, NULL, NULL);
	// Run the test loop
	//float* test_array = (float*)malloc(COPY_SIZE * sizeof(cl_float));


	cl_event e[NUM_LOOPS_GROUP + 1];

	clock_t start, end;
	double time_elapsed;

	printf("\nStarting computation\n");

	// First version of the loop, without reading the output buffer
	start = clock();
	for (int i = 0; i < NUM_LOOPS; i++)
	{
		clEnqueueNDRangeKernel(queue, kernel_vector_sum, 1, &global_work_offset, &global_work_size, &local_work_size, 1, NULL, NULL);
		clEnqueueNDRangeKernel(queue, kernel_vector_modulo, 1, &global_work_offset, &global_work_size, &local_work_size, 1, NULL, NULL);
	}

	clFinish(queue);
	end = clock();

	time_elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("\nTime elapsed without read: %f\n", time_elapsed);


	// Second version of the loop, with reading the buffer
	// Commands are chained with events, to avoid blocking reads
	start = clock();
	clEnqueueNDRangeKernel(queue, kernel_vector_sum, 1, &global_work_offset, &global_work_size, &local_work_size, 1, NULL, &e[0]);
	int j = 0;
	for (int i = 0; i < NUM_LOOPS; i++, j++)
	{
		clEnqueueReadBuffer(queue, c_mem, CL_FALSE, 0, COPY_SIZE * sizeof(cl_float), test_array, 0, NULL, &e[j]);
		clEnqueueNDRangeKernel(queue, kernel_vector_modulo, 1, &global_work_offset, &global_work_size, &local_work_size, 1, &e[j], NULL);
		clEnqueueNDRangeKernel(queue, kernel_vector_sum, 1, &global_work_offset, &global_work_size, &local_work_size, 1, NULL, NULL);

		if (i % NUM_LOOPS_GROUP == 0)
		{
			clFinish(queue);
			j = 0;
		}
	}

	clFinish(queue);
	end = clock();

	time_elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Time elapsed with read: %f\n", time_elapsed);

	clEnqueueUnmapMemObject(queue, tmp_buf, test_array, 0, NULL, NULL);

	return 0;
}

*/
const char *getErrorString(cl_int error){
    switch (error)
    {
    // run-time and JIT compiler errors
    case 0:
        return "CL_SUCCESS";
    case -1:
        return "CL_DEVICE_NOT_FOUND";
    case -2:
        return "CL_DEVICE_NOT_AVAILABLE";
    case -3:
        return "CL_COMPILER_NOT_AVAILABLE";
    case -4:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5:
        return "CL_OUT_OF_RESOURCES";
    case -6:
        return "CL_OUT_OF_HOST_MEMORY";
    case -7:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8:
        return "CL_MEM_COPY_OVERLAP";
    case -9:
        return "CL_IMAGE_FORMAT_MISMATCH";
    case -10:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11:
        return "CL_BUILD_PROGRAM_FAILURE";
    case -12:
        return "CL_MAP_FAILURE";
    case -13:
        return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14:
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15:
        return "CL_COMPILE_PROGRAM_FAILURE";
    case -16:
        return "CL_LINKER_NOT_AVAILABLE";
    case -17:
        return "CL_LINK_PROGRAM_FAILURE";
    case -18:
        return "CL_DEVICE_PARTITION_FAILED";
    case -19:
        return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30:
        return "CL_INVALID_VALUE";
    case -31:
        return "CL_INVALID_DEVICE_TYPE";
    case -32:
        return "CL_INVALID_PLATFORM";
    case -33:
        return "CL_INVALID_DEVICE";
    case -34:
        return "CL_INVALID_CONTEXT";
    case -35:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case -36:
        return "CL_INVALID_COMMAND_QUEUE";
    case -37:
        return "CL_INVALID_HOST_PTR";
    case -38:
        return "CL_INVALID_MEM_OBJECT";
    case -39:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40:
        return "CL_INVALID_IMAGE_SIZE";
    case -41:
        return "CL_INVALID_SAMPLER";
    case -42:
        return "CL_INVALID_BINARY";
    case -43:
        return "CL_INVALID_BUILD_OPTIONS";
    case -44:
        return "CL_INVALID_PROGRAM";
    case -45:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46:
        return "CL_INVALID_KERNEL_NAME";
    case -47:
        return "CL_INVALID_KERNEL_DEFINITION";
    case -48:
        return "CL_INVALID_KERNEL";
    case -49:
        return "CL_INVALID_ARG_INDEX";
    case -50:
        return "CL_INVALID_ARG_VALUE";
    case -51:
        return "CL_INVALID_ARG_SIZE";
    case -52:
        return "CL_INVALID_KERNEL_ARGS";
    case -53:
        return "CL_INVALID_WORK_DIMENSION";
    case -54:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case -55:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case -56:
        return "CL_INVALID_GLOBAL_OFFSET";
    case -57:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case -58:
        return "CL_INVALID_EVENT";
    case -59:
        return "CL_INVALID_OPERATION";
    case -60:
        return "CL_INVALID_GL_OBJECT";
    case -61:
        return "CL_INVALID_BUFFER_SIZE";
    case -62:
        return "CL_INVALID_MIP_LEVEL";
    case -63:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64:
        return "CL_INVALID_PROPERTY";
    case -65:
        return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66:
        return "CL_INVALID_COMPILER_OPTIONS";
    case -67:
        return "CL_INVALID_LINKER_OPTIONS";
    case -68:
        return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000:
        return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001:
        return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002:
        return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003:
        return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004:
        return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005:
        return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default:
        return "Unknown OpenCL error";
    }
}




int GetPlatformInfo(std::vector<cl::Platform> platforms, std::vector<cl::Device> devices){

       if(platforms.size() == 0){
            std::cerr << "No platforms found" << std::endl;
            return -1;   
        }

        std::cout << "Number of available OpenCL platforms: " << platforms.size() << std::endl << std::endl;

    for (auto p : platforms){
        p.getInfo((cl_platform_info) CL_PLATFORM_NAME, &tmpstr);
        std::cout << "Platform's name: " << tmpstr << "\n";

        p.getInfo((cl_platform_info) CL_PLATFORM_VENDOR, &tmpstr);
        std::cout << "Platform's vendor: " << tmpstr << "\n";

        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);


        std::cout << "Number of computing devices: " << devices.size() << "\n";
        std::cout << std::endl;

        for(auto d : devices){
            d.getInfo((cl_device_info)CL_DEVICE_NAME, &tmpstr);
            std::cout << "Device's name: " << tmpstr << "\n";

            d.getInfo((cl_device_info)CL_DEVICE_VENDOR, &tmpstr);
            std::cout << "Device's vendor: " << tmpstr << "\n";

            d.getInfo(CL_DEVICE_TYPE, &device_type);

            if(device_type & CL_DEVICE_TYPE_CPU)
                std::cout << "Device's type: CPU." << "\n";

            if(device_type & CL_DEVICE_TYPE_GPU)
                std::cout << "Device's type: GPU." << "\n";

            if(device_type & CL_DEVICE_TYPE_ACCELERATOR)
                std::cout << "Device's type: ACCELERATOR." << "\n";

            d.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &tmp);
            std::cout << "Computing units: " << tmp << "\n";

            d.getInfo(CL_DEVICE_IMAGE_SUPPORT, &bool_tmp);
            std::cout << "Image support: " << ((bool_tmp == CL_TRUE) ? std::string("yes") : std::string("no")) << "\n";

            
        }
        std::cout << std::endl;
    }

    return 0;
}

void printArray(std::string header, const int *vec){
    std::cout << "\n" << header << "\n";

    for(int i = 0; i < N; i++){
        
        std::cout << vec[i] << " ";
        if( i % 3 == 2)
            std::cout << "\n";
         
    }
    std::cout << "\n";
}


void fillArray(float *vec, int _seed){
    int seed = (unsigned int) time(NULL) % _seed;

    srand(seed);

    for(int i = 0; i < N; i++){
        vec[i] = float(10*rand()/(RAND_MAX + 1.0));
    }
}

void CreateMatrix(std::array<std::array<int, N>, N>& matrix, int num){ // tworzy macierz sasiedzctwa     
     
    for (int i = 0; i < N; i++ ){      
        for(int j = 0; j < N; j++){            
                matrix[i][j] = num;
                std::cout << matrix[i][j] << " ";            
            }
        std::cout << std:: endl;  
        }        
}

void PrintStdArray(const std::array<std::array<int, N>, N> &array){
    std::cout << "length: " << array.size() << std::endl;
    
    for (int i = 0; i < array.size(); i++ ){
        //std::cout << "n[" << j << "] = " << n[j] << std::endl;
        for(int j = 0; j < array.size(); j++){
            std::cout << array[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


