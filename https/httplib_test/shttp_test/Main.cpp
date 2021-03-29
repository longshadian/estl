#if 1

#include <iostream>
#include <string>

#include "SHttpServer.h"

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    SHttpServer s;
    s.Init("0.0.0.0", 10086);
    s.Loop();

    return 0;
}
#endif

