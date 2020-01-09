#include <iostream>
#include <string>

int TestConstPtr();
int TestRegex();
int TestFilesystem();

int main(int argc, char** argv)
{
    //TestConstPtr();
    //TestRegex();
    TestFilesystem();

#if defined (WIN32)
    system("pause");
#endif

    return 0;
}

