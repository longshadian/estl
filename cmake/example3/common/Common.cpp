#include "common/Common.h"

#include <sstream>

std::string Common::ToString(int v)
{
    std::ostringstream ostm;
    ostm << v;
    return ostm.str();
    //return std::to_string(v);
}

