#include <iostream>
#include <string>

int TestConstPtr();

int main(int argc, char** argv)
{
    TestConstPtr();

#if defined (WIN32)
    system("pause");
#endif

    return 0;
}

