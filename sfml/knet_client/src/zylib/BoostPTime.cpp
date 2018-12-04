#include "zylib/BoostPTime.h"

#include <sstream>

namespace zylib {

std::string formatPTime(boost::posix_time::ptime t, const char* fmt)
{
    try {
        std::ostringstream ostm{};
        if (fmt) {
            boost::posix_time::time_facet* facet = new boost::posix_time::time_facet(fmt);
            ostm.imbue(std::locale(std::locale(), facet));
        }
        ostm << t;
        return ostm.str();
    } catch (...) {
        return {};
    }
}

}
