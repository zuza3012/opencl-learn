#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#define CL_HPP_TARGET_OPENCL_VERSION 200 //dobre rozwiazanie jak wiesz na jakim kompie to ma isc
//#define CL_USE_DEPRECATED_OPENCL_2_0_API // do clCreateCommandQueue bo opencl 2.0 nie wspiera chyba
//#define __NO_STD_VECTOR  - z ta kupa nie ciaua

#define __MAX_DEFAULT_VECTOR_SIZE 128

#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/cl.hpp>
#else 
    #include <CL/cl2.hpp>
#endif


//https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes/24336429#24336429
const char *getErrorString(cl_int error){
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}



const char* kernelSource = "__kernel void helloDecode(__global char *inTxt, __global char *outTxt, int N){\n"
"   int idx = get_global_id(0);\n"
"   if(idx < N){\n"
"       outTxt[idx] = inTxt[idx] - 5;\n"
"   }\n"
"}\n"
"\n";

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

cl_platform_id *opencl_platforms = NULL;
cl_device_id *platform_devices = NULL;

cl_context cl_compute_context = NULL;

cl_command_queue cl_compute_command_queue = NULL;

cl_mem device_in_txt; // bufor
cl_mem device_out_txt; // bufor

cl_program cl_kernel_program;

cl_kernel kernel = NULL;
size_t globalWorkSize;

int main(int argc, char* argv[]){
    // szyfrujemy szyfrem cezara
    //zero na koncu to NULL (ze koniec linii)
    char host_int_txt[] = {'H' + 5, 'e' + 5, 'l' + 5, 'l' + 5, 'o' + 5, ' ' + 5, 'W' + 5, 'o' + 5, 'r' + 5, 'l' + 5, 'd' + 5, '!' + 5, 0};

    char host_out_txt[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    const int N = (sizeof(host_int_txt) / sizeof(char) - 1);
    size_t dataBytes = sizeof(char) * N;
    cli_err_num = CL_SUCCESS;

    cli_err_num = clGetPlatformIDs(0, NULL, &clui_num_platforms);// zwraca ilosc platfrom dostepnych m.in.

    if(cli_err_num != CL_SUCCESS){
        std::cout << "OpenCL access denied!" << std::endl;
        return -1;
    }

    std::cout << "Platforms available: " << clui_num_platforms << std::endl;

    opencl_platforms = (cl_platform_id*)malloc(clui_num_platforms * sizeof(cl_platform_id)); // zwraca wskaznik na pocztek bloku pamieci

    if(opencl_platforms == NULL){
        std::cout << "Could not allocate memory for OpenCL's platforms" << std::endl;
        return -1;
    }

    cli_err_num = clGetPlatformIDs(clui_num_platforms, opencl_platforms, NULL); // teraz mamy opencl_platforms

    if(cli_err_num != CL_SUCCESS){
        free((void*)opencl_platforms); // zwracamy pamiec
        opencl_platforms = NULL;
        return -1;
    }

    cli_err_num = clGetDeviceIDs(opencl_platforms[0],CL_DEVICE_TYPE_ALL, 0,NULL, &clui_num_devices);

    if(cli_err_num != CL_SUCCESS){
        free((void *)opencl_platforms); // zwalniamy pamiec zamallocowanom
        opencl_platforms = NULL;
        std::cout << "Error in reading number of computing devices" << std::endl;
        return -1;
    }

    std::cout << "Available computing devices: " << clui_num_devices << std::endl;

    platform_devices = (cl_device_id*)malloc(clui_num_devices * sizeof(cl_device_id));

    cli_err_num = clGetDeviceIDs(opencl_platforms[0],CL_DEVICE_TYPE_ALL, clui_num_devices, platform_devices, NULL);

    if(cli_err_num != CL_SUCCESS){
        std::cout << "Could not read about devices!" << std::endl;
        return EXIT_FAILURE;
    }
    cl_compute_context = clCreateContext(NULL, clui_num_devices, &platform_devices[0],NULL, NULL, &cli_err_num);

    //cl_compute_command_queue = clCreateCommandQueue(cl_compute_context, platform_devices[0],0, &cli_err_num);
    //cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_compute_command_queue = clCreateCommandQueueWithProperties(cl_compute_context,platform_devices[0],0 , &cli_err_num);
    

    if(cli_err_num != CL_SUCCESS)
        std::cerr << getErrorString(cli_err_num) << std::endl;


    /*if(cl_compute_command_queue != CL_SUCCESS){
        std::cout << "Could not open command queue!" << std::endl;
        return EXIT_FAILURE;
    }*/

    device_in_txt = clCreateBuffer(cl_compute_context, CL_MEM_READ_ONLY, dataBytes, NULL, &cli_err_num);
    device_out_txt = clCreateBuffer(cl_compute_context, CL_MEM_WRITE_ONLY, dataBytes, NULL, &cli_err_num);

    cli_err_num = clEnqueueWriteBuffer(cl_compute_command_queue, device_in_txt, CL_TRUE, 0, dataBytes, &host_int_txt[0], 0, NULL, NULL);

    cl_kernel_program = clCreateProgramWithSource(cl_compute_context,1,(const char**)&kernelSource, NULL, &cli_err_num);

    if(!cl_kernel_program){
        std::cout << "Could not create kernel program!" << std::endl;
        std::cout << "Error code: " << cli_err_num <<std::endl;
        return EXIT_FAILURE;
    }

    cli_err_num = clBuildProgram(cl_kernel_program, clui_num_devices,platform_devices, NULL, NULL, NULL);

    kernel = clCreateKernel(cl_kernel_program, "helloDecode", &cli_err_num);

    if(!kernel || cli_err_num != CL_SUCCESS){
        std::cout << "Could not create kernel program!" << std::endl;
        exit(1);
    }

    cli_err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_in_txt);
    cli_err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_out_txt);
    cli_err_num |= clSetKernelArg(kernel, 2, sizeof(int), &N);

    globalWorkSize = N;
    
    cli_err_num = clEnqueueNDRangeKernel(cl_compute_command_queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);

    if(cli_err_num){
        std::cout << "Could not open kernel!" << std::endl;
        return EXIT_FAILURE;
    }

    cli_err_num = clEnqueueReadBuffer(cl_compute_command_queue,device_out_txt, CL_TRUE, 0, dataBytes, host_out_txt,0, NULL, NULL);

    if(cli_err_num != CL_SUCCESS){
        std::cout << "Could not read output!" << std::endl;
        exit(1);
    }

    std::cout << "Read data: " << host_out_txt  << std::endl;
  

    clReleaseKernel(kernel);
    clReleaseProgram(cl_kernel_program);
    clReleaseCommandQueue(cl_compute_command_queue);
    clReleaseMemObject(device_out_txt);
    clReleaseMemObject(device_in_txt);
    clReleaseContext(cl_compute_context);

    free((void *)platform_devices);
    free((void *) opencl_platforms);

    return 0;
}