#include <ctime>
#include <fstream>
#include <iostream>
#include <exception>
#include <stdio.h>


#include <iomanip>
#include <array>
#include <stdio.h>
#include <vector>
#include <cblas.h>

#define CL_HPP_TARGET_OPENCL_VERSION 110

#define __CL_ENABLE_EXCEPTIONS
#define __MAX_DEFAULT_VECTOR_SIZE 128

#if defined(__APPLE__) || defined(MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/opencl.h>  
   #include <CL/cl.hpp>   
#endif

#define len 1024
float buf_x[len]__attribute__((aligned(16)));

const int N = 5;
const int size = 5;


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

//https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes/24336429#24336429
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
/*
void CreateMatrix(std::vector< std::vector<int> >& matrix, int N, int num){ // tworzy macierz sasiedzctwa     
    matrix.resize(N);// rows
    for( std::vector<std::vector<int> >::iterator it = matrix.begin(); it != matrix.end(); ++it){
        it->resize(N);// cols
    }    
    for (int i = 0; i < N; i++ ){      
        for(int j = 0; j < N; j++){            
                matrix[i][j] = num;
                std::cout << matrix[i][j] << " ";            
            }
        std::cout << std:: endl;  
        }        
}
*/
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


int main(int argc, char* argv[]){
    size_t datasize = N * sizeof(float);

    const int lda = 3, ldb = 3, ldc = 3;
    int m, n, k;
    float alpha, beta;

    float a[] = { 0.00, 1.00, 1.00,
                    1.00, 0.00, 1.00,
                    1.00, 1.00, 0.0 };

    float b[] = { 1, 0, 0,
                    1, 0, 0,
                    -1, 0, 0 };

    float c[] = { 0.00, 0.00, 0,00,
                    0.00, 0.00, 0,00,
                    0.00, 0.00, 0,00 };

    m = 3; n = 3; k = 3;

    alpha = 1.0; beta = 0.0;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

    printf ("[ %g, %g, %g\n", c[0], c[1], c[2]);
    printf ("  %g, %g, %g\n", c[3], c[4], c[5]);
    printf ("  %g, %g, %g ]\n", c[6], c[7], c[8]);

    std::array<std::array<int, N>, N> matrixA, matrixB, matrixC;
  
   int widthA = 5, widthB = 5, heightA = 5, heightB = 5;

    for (int i = 0; i < N; i++ ){     // rows
        for(int j = 0; j < N; j++){    //   cols     
            matrixA[i][j] = 1;
            if(i >= 3)
                matrixA[i][j] = 0;
            std::cout << matrixA[i][j] << " ";            
        
        }
        std::cout << std:: endl;  
    }        
    std::cout << std::endl;

    for (int i = 0; i < N; i++ ){      
        for(int j = 0; j < N; j++){            
            matrixB[i][j] = 2;
            if(j>= 4)
                matrixB[i][j] = 0;
            std::cout << matrixB[i][j] << " ";            
        }
        std::cout << std:: endl;  
    }        

    std::cout << std::endl;
    for (int i = 0; i < N; i++ ){      
        for(int j = 0; j < N; j++){            
            matrixC[i][j] = 0;
            
            std::cout << matrixC[i][j] << " ";            
        }
        std::cout << std:: endl;  
    }        
    
    

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
      
        cl::Buffer bufferC(context, CL_MEM_READ_ONLY, N * N* sizeof(float)); // matrix C
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY, N * N * sizeof(float)); // matrix B
        cl::Buffer bufferA(context, CL_MEM_READ_WRITE, N * N * sizeof(float)); // matrix A
 

        queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, N * N * sizeof(float), matrixA.data());
        queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, N * N* sizeof(float), matrixB.data());
            
        
           
        std::string pwd_path = getenv("PWD");
        std::string file_rel_path = "/../dot.cl";

        std::string path = pwd_path+file_rel_path;
    

        std::ifstream sourceFile(path);
        std::string sourceCode(std::istreambuf_iterator <char>(sourceFile),(std::istreambuf_iterator <char>()));

        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
       
        program = cl::Program(context, source);

        program.build(devices);

        cl::Kernel kernel(program, "dot_product");
       
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, widthA);
        kernel.setArg(4, widthB);
        kernel.setArg(5, heightA);
        kernel.setArg(6, heightB);

        

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N, N));//, local);//, local); ; 


        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, N * N * sizeof(float), matrixC.data());// zmienione
         std::cout << "Dot:" << std::endl;
        PrintStdArray(matrixC);
/*
        for(int i = 0; i < 16; i++){
                
            if ( i %4 == 0)
                std::cout << std::endl;

            std::cout << matrixC[i]<< std::endl;
                
                
                
        }
*/
     
        /*
        for(int i = 0; i < 4; i++){
                for(int j = 0; j < 4; j++){
                    std::cout << matrixC[i][j]<< std::endl;
                }
                std::cout << std::endl;
                
        }
        
        */
        
            
        


       
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