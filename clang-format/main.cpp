
#include <string>
#include <iostream>

namespace n1
{

struct S
{
    S(): i{},j{}
    {}

int i;
std::string j;
};

class X
{
    public:
        X();

        void F(std::string a1, std::string* a2, const std::string& a3, std::string a4)
        {

        }
};

class Y : 
public X
{
    public : Y() {}
};

static inline
void f(int * arg1, int * arg2, int* arg3)
{
    int n = 0;
    int * pn = &n;
    const int& rn = n;
    const char* pstr = "01234567890012345678900123456789001234567890012345678900123456789001234567890012345678900123456789001234567890";
    const char* pstr2 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            ;

    if (n > 0) {

    }

    while (1) {
        int a = 0;
    }

    do {
            int a =0;
    } while (0);

    for (int i = 0; i != 100; ++i) {
        int n = 0;
    }

    int a{};

    switch (a)
 {
     case 1: {
        std::cout << 1111 << "\n";
        break;
     }
     case 2: {
         break;
     }
     default: {
         std::cout << 1111 << "\n";
     }
 }

 try {

 } catch(std::overflow_error& e1) {
     std::cout << e1.what() << "\n";
 } catch (std::runtime_error& e2) {
std::cout << e2.what() << "\n";
 } catch (...) {

 }

 {
     int block1{};
     int * block2  = &block1;
 }
}
    
}



int main()
{
    int f = 0;


    int j = 10;

n1::f(nullptr, nullptr, nullptr);
}



