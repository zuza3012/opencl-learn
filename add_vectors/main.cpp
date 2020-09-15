#include <ctime>
#include <fstream>
#include <iostream>
#include <exception>

//#define CL_HPP_MINIMUM_OPENCL_VERSION 120
//#define CL_HPP_TARGET_OPENCL_VERSION 120 //dobre rozwiazanie jak wiesz na jakim kompie to ma isc
//#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_TARGET_OPENCL_VERSION 110


//#define CL_USE_DEPRECATED_OPENCL_2_0_API // do clCreateCommandQueue bo opencl 2.0 nie wspiera chyba
//#define __NO_STD_VECTOR // - z ta kupa nie ciaua
//#define CL_HPP_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#define __MAX_DEFAULT_VECTOR_SIZE 128

#if defined(__APPLE__) || defined(MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/opencl.h>  
   #include <CL/cl.hpp>
   // #include <CL/cl2.hpp>
    
#endif

const int N = 16;


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

const char* kernelSource = 
"__kernel void sum(\n"
"__global const float *a, __global const float *b,\n"
"__global float *c){\n"
"\n"
"int id = get_global_id(0);\n"
"c[id] = a[id] + b[id];\n"
"\n"
"}\n"
"\n";

int i;

cl_int cli_err;


void printArray(std::string header, const int *vec){
    std::cout << "\n" << header << "\n";

    for(int i = 0; i < N; i++){
        std::cout << vec[i] << " ";
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

int main(int argc, char* argv[]){
    size_t datasize = N * sizeof(int);
    
    // nie robić tak: int *A;, a potem w mainie: A = new int[N]!!!!!
    int *A = new int[N];
    int *B = new int[N];
    int *C = new int[N];
    

    for(int i = 0; i < N; i++){
        A[i] = i;
        B[i] = i;
    }
    printArray("vector A:", A);
    printArray("vector B:", B);


    try{
        std::vector<cl::Platform> platforms; // to musi byc tutaj!!!
        cl::Platform::get(&platforms);

        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
            
        cl::Context context(devices);

        cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

        cl::Buffer bufferA(context, CL_MEM_READ_ONLY, datasize);
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY, datasize);
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, datasize);

        queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, datasize, A);
        queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, datasize, B);

        std::string pwd_path = getenv("PWD");
        std::string file_rel_path = "/../add_vectors.cl";

        std::string path = pwd_path+file_rel_path;
    

        std::ifstream sourceFile(path);
        std::string sourceCode(std::istreambuf_iterator <char>(sourceFile),(std::istreambuf_iterator <char>()));

        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
       //cl::Program::Sources source(1, std::make_pair(kernelSource, strlen(kernelSource)));
        
        

        // program + build
        cl::Program program = cl::Program(context, source);

        program.build(devices);

        cl::Kernel kernel(program, "sum");


        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);

        cl::NDRange global(N);
        cl::NDRange local(4);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local); // kernel, offset, global, local, vektor z eventami, wsk na eventy

        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, datasize, C);

        printArray("output:", C);


       
    }catch(cl::Error e){
         std::cout << e.what() << ": Error code " << e.err() <<  getErrorString(e.err()) << std::endl;

    }

        //program.build(devices);
        /*
        try{
            
        }
        catch (cl::Error& e2){
            if (e2.err() == CL_BUILD_PROGRAM_FAILURE){
                for (cl::Device dev : devices){
                // Check the build status
                cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                if (status != CL_BUILD_ERROR)
                    continue;

                // Get the build log
                std::string name     = dev.getInfo<CL_DEVICE_NAME>();
                std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                std::cerr << "Build log for " << name << ":" << std::endl
                            << buildlog << std::endl;
                }
            }
            else{
                throw e2;
            }
        }
        */
        


        //cl::Event event;
        // wywolanie jadra obliczeniowego
       
        //event.wait();

        //odczytanie danych z buffera wyjsciowego
        
        /*
        clReleaseMemObject(device_in_a());
        8clReleaseMemObject(device_in_b());
        clReleaseMemObject(device_out_c());
        */
        
       
       
/*
    }catch(cl::Error e){

        std::cout << e.what() << ": Error code " << e.err() <<  getErrorString(e.err()) << std::endl;


    }
    */
    //delete host_in_a;
    //delete host_in_b;
   // delete host_out_c;

    
    return 0;
}