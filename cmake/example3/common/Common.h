#pragma once

#ifdef WIN32
 #define DllExport   __declspec( dllexport )//�궨��
#else
 #define DllExport
#endif

#include <string>

class DllExport Common
{
public:
    static std::string ToString(int v);
    
};