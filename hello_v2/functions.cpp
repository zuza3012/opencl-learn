#include "functions.h"
#include <fstream>
#include <string>
//#include <streambuf>
#include <iostream>
#include <sstream>

const char * readFile(const char * filename){



    std::ifstream file(filename, std::ios::in);
    if (!file.is_open()){
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << file.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    
    
    std::cout << srcStdStr << std::endl;

    return srcStdStr.c_str();
}