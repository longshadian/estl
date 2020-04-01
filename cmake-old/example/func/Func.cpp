#include "func/Func.h"

#include <string>
#include <iostream>

#include "func/ext/Foo.h"

int func(int v)
{
    return v * 2 + foo(v);
}