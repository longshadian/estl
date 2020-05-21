#include "Common.h"

namespace common
{

std::string errno_to_string(int eno)
{
    std::ostringstream ostm{};
    ostm << eno << " ";
    if (eno == ERANGE)
        ostm << "ERANGE";
    return ostm.str();
}

} // namespace common

