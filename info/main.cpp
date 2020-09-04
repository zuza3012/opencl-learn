#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
//#define CL_HPP_TARGET_OPENCL_VERSION 200 dobre rozwiazanie jak wiesz na jakim kompie to ma isc

//#define __NO_STD_VECTOR  - z ta kupa nie ciaua
#define __MAX_DEFAULT_VECTOR_SIZE 128

#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/cl.hpp>
#else 
    #include <CL/cl2.hpp>
#endif



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

cl::vector<cl::Platform> platforms;
cl::vector<cl::Device> devices;

cl::vector<cl::ImageFormat> image_formats;

const char* imgFormatToChars(cl_uint imgFormatCode){
    if(imgFormatCode == CL_R)   return "CL_R";
    if(imgFormatCode == CL_A)   return "CL_A";
    if(imgFormatCode == CL_RG)   return "CL_RG";
    if(imgFormatCode == CL_RA)   return "CL_RA";
    if(imgFormatCode == CL_RGB)   return "CL_RGB";
    if(imgFormatCode == CL_RGBA)   return "CL_RGBA";
    if(imgFormatCode == CL_BGRA)   return "CL_BGRA";
    if(imgFormatCode == CL_ARGB)   return "CL_ARGB";
    if(imgFormatCode == CL_INTENSITY)   return "CL_INTENSITY";
    if(imgFormatCode == CL_LUMINANCE)   return "CL_LUMINANCE";

    if(imgFormatCode == CL_SNORM_INT8)   return "CL_SNORM_INT8";
    if(imgFormatCode == CL_SNORM_INT16)   return "CL_SNORM_INT16";
    if(imgFormatCode == CL_UNORM_INT8)   return "CL_UNORM_INT8";
    if(imgFormatCode == CL_UNORM_INT16)   return "CL_UNORM_INT16";
    if(imgFormatCode == CL_UNORM_SHORT_555)   return "CL_UNORM_SHORT_555";
    if(imgFormatCode == CL_UNORM_INT_101010)   return "CL_UNORM_INT_101010";

    if(imgFormatCode == CL_SIGNED_INT8)   return "CL_SIGNED_INT8";
    if(imgFormatCode == CL_SIGNED_INT16)   return "CL_SIGNED_INT16";
    if(imgFormatCode == CL_SIGNED_INT32)   return "CL_SIGNED_INT32";

    if(imgFormatCode == CL_UNSIGNED_INT8)   return "CL_UNSIGNED_INT8";
    if(imgFormatCode == CL_UNSIGNED_INT16)   return "CL_UNSIGNED_INT16";
    if(imgFormatCode == CL_UNSIGNED_INT32)   return "CL_UNSIGNED_INT32";


    if(imgFormatCode == CL_HALF_FLOAT)   return "CL_HALF_FLOAT";
    if(imgFormatCode == CL_FLOAT)   return "CL_FLOAT";

    return "not defined format";
    
}

int main(int argc, char* argv[]){

    cl::Platform::get( &platforms );
    std::cout << "Number of available OpenCL platforms: " << platforms.size() << std::endl << std::endl;

    for (auto p : platforms){
        p.getInfo((cl_platform_info) CL_PLATFORM_NAME, &tmpstr);
        std::cout << "Platform's name: " << tmpstr << "\n";

        p.getInfo((cl_platform_info) CL_PLATFORM_VENDOR, &tmpstr);
        std::cout << "Platform's vendor: " << tmpstr << "\n";

        p.getInfo((cl_platform_info) CL_PLATFORM_EXTENSIONS, &tmpstr);
        std::cout << "Extentions: " << tmpstr << "\n";

        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);


        std::cout << "Number of computing devices: " << devices.size() << "\n";
        std::cout << std::endl;

        for(auto d : devices){
            d.getInfo((cl_device_info)CL_DEVICE_NAME, &tmpstr);
            std::cout << "Device's name: " << tmpstr << "\n";

            d.getInfo((cl_device_info)CL_DEVICE_VENDOR, &tmpstr);
            std::cout << "Device's vendor: " << tmpstr << "\n";

            d.getInfo((cl_device_info)CL_DEVICE_EXTENSIONS, &tmpstr);
            std::cout << "Extentions: " << tmpstr << "\n";

            d.getInfo(CL_DEVICE_TYPE, &device_type);

            if(device_type & CL_DEVICE_TYPE_CPU)
                std::cout << "Device's type: CPU." << "\n";

            if(device_type & CL_DEVICE_TYPE_GPU)
                std::cout << "Device's type: GPU." << "\n";

            if(device_type & CL_DEVICE_TYPE_ACCELERATOR)
                std::cout << "Device's type: ACCELERATOR." << "\n";

            d.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &tmp);
            std::cout << "Computing units: " << tmp << "\n";

            d.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &tmp);
            std::cout << "DEVICE_MAX_WORK_GROUP_SIZE: " << tmp << "\n";

            d.getInfo(CL_DEVICE_IMAGE_SUPPORT, &bool_tmp);
            std::cout << "Image support: " << ((bool_tmp == CL_TRUE) ? std::string("yes") : std::string("no")) << "\n";

            d.getInfo(CL_DEVICE_SINGLE_FP_CONFIG, &dfpc_tmp);
            std::cout << "Single fp: " << ((dfpc_tmp & CL_FP_DENORM) ? "denorm" : "" )
            << ((dfpc_tmp & CL_FP_FMA) ? "fma" : "" ) << ((dfpc_tmp & CL_FP_INF_NAN) ? "INF-quietNaNs" : "" )
            << ((dfpc_tmp & CL_FP_ROUND_TO_INF) ? "round-to-inf" : "" ) << ((dfpc_tmp & CL_FP_ROUND_TO_NEAREST) ? "round-to-nearest" : "" )
            << ((dfpc_tmp & CL_FP_ROUND_TO_ZERO) ? "round-to-zero" : "" ) << std::endl;

            cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0](),0};
            cl::Context context(CL_DEVICE_TYPE_GPU, properties);

            image_formats.clear();

            context.getSupportedImageFormats(CL_MEM_READ_ONLY,CL_MEM_OBJECT_IMAGE2D, &image_formats);

            int i = 1;

            for(auto iformat : image_formats){
                std::cout << "\t\t" << i << std::setw(17) << imgFormatToChars(iformat.image_channel_order) 
                << std::setw(23) << imgFormatToChars(iformat.image_channel_data_type) << std::endl;

                i++;

            }
        }
        std::cout << std::endl;
    }
    
    
    getchar();

    return 0;
}
