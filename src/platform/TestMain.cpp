#include <iostream>
#include <string>

#include "Common.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include "../doctest/doctest.h"

int main(int argc, char** argv)
{
    PrintInfo("Test Main argc: %d", argc);
#ifdef NDEBUG
    std::cout << "----------------------release\n";
#else
    std::cout << "----------------------debug\n";
#endif

    doctest::Context context;
    context.applyCommandLine(argc, argv);

    int res = context.run(); // run doctest
    // important - query flags (and --exit) rely on the user doing this
    if (context.shouldExit()) {
        // propagate the result of the tests
        return res;
    }
    return 0;
}

