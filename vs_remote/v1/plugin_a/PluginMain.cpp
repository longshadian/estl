
#include <string>
#include <iostream>
#include <any>

#include <boost/dll.hpp>

#include "plugin_a/GlobalInstance.h"

static void plugin_a_load() 
#if defined (__GNUC__)
__attribute__((constructor))
#endif
;

static void plugin_a_unload()
#if defined (__GNUC__)
 __attribute__((destructor))
#endif
;

void plugin_a_load()
{
    std::cout << "plugin a load\n";
}

void plugin_a_unload()
{
    std::cout << "plugin a unload\n";
}

extern "C" 
{
    BOOST_SYMBOL_EXPORT std::string name();
    BOOST_SYMBOL_EXPORT std::string version();
    BOOST_SYMBOL_EXPORT std::string desc();
}

std::string name() 
{
    return "blacklist";
}

std::string version()
{
    return "0.1";
}

std::string desc()
{
    return plugin::a::GlobalInstance::get()->execute();
}

