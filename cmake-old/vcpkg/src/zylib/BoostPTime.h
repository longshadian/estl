#pragma once

#include <string>
//#include <sstream>

#include <boost/date_time.hpp>
#include <boost/date_time/c_local_time_adjustor.hpp>

#include "zylib/ZylibDefine.h"

namespace zylib {

inline
boost::posix_time::ptime utcToLocal(time_t t)
{
    return boost::date_time::c_local_adjustor<boost::posix_time::ptime>::utc_to_local(
        boost::posix_time::from_time_t(t));
}

std::string ZYLIB_EXPORT formatPTime(boost::posix_time::ptime t, const char* fmt = nullptr);

inline
boost::posix_time::ptime local_time() 
{ 
	return boost::posix_time::microsec_clock::local_time();
}

inline
boost::posix_time::ptime universal_time()
{ 
	return boost::posix_time::microsec_clock::universal_time();
}

}
