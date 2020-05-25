#pragma once

#if defined (__GNUC__)
    #if __GNUC__ < 8
        #include <experimental/filesystem>
        namespace fs = std::experimental::filesystem;
    #else
        #include <filesystem>
        namespace fs = std::filesystem;
    #endif
#else
    #include <filesystem>
    namespace fs = std::filesystem;
#endif

