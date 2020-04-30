#include <iostream>
#include <algorithm>
#include <string>
#include <stlsoft/string/string_tokeniser.hpp>

static int fun()
{
  stlsoft::string_tokeniser<std::string, char> tokens(":abc::def:ghi:jkl::::::::::", ':');

  std::copy(tokens.begin(), tokens.end(), std::ostream_iterator<std::string>(std::cout, " "));
  return 0;
}

void TestStringTokeniser()
{
    fun();
}

