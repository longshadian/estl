#include <iostream>
#include <string>

#define WELL_NUMBER(s) s 

#define GCC_IGNORE_WARNING(name, warning) \
WELL_NUMBER(#)##pragma GCC diagnostic push \
WELL_NUMBER(#)##pragma GCC diagnostic ignored warning \
#include name \
WELL_NUMBER(#)##pragma GCC diagnostic pop 

GCC_IGNORE_WARNING("a.h", "-Wunused-parameter")

/*
#define GCC_IGNORE_WARNING(name, warning) \
#pragma GCC diagnostic push \
#pragma GCC diagnostic ignored warning \
#include name \
#pragma GCC diagnostic pop 

GCC_IGNORE_WARNING("a.h", "-Wunused-parameter")
*/

//#include "a.h"

int main()
{
    fun(12);
    return 0;
}
