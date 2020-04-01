#pragma once

#include <boost/log/trivial.hpp>

#define LOG(x) BOOST_LOG_TRIVIAL(x) << "[" << __LINE__ << ":" << __FUNCTION__ << "] "
