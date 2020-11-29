#include <iostream>

void Fun1(char buf1[10], char buf2[], char (&buf3)[1])
{
    std::cout << "sizeof buf[10]: " << sizeof(buf1) << "\n";
    std::cout << "sizeof buf[]: " << sizeof(buf2) << "\n";
    std::cout << "sizeof (&buf)[10]: " << sizeof(buf3) << "\n";
}

template <typename T>
struct Array;

template <typename T, size_t SZ>
struct Array<T[SZ]>
{
    static void print()
    {
        std::cout << "T[" << SZ << "] sizeof: " << sizeof(T[SZ]) << "\n";
    }
};

template <typename T, size_t SZ>
struct Array<T(&)[SZ]>
{
    static void print()
    {
        std::cout << "T(&)[" << SZ << "] sizeof: " << sizeof(T(&)[SZ]) << "\n";
    }
};

template <typename T>
struct Array<T[]>
{
    static void print()
    {
        std::cout << "T[] sizeof: " << sizeof(T[]) << "\n";
    }
};

template <typename T>
struct Array<T(&)[]>
{
    static void print()
    {
        //std::cout << "T(&)[] sizeof: " << sizeof(T(&)[]) << "\n";
        std::cout << "T(&)[]\n";
    }
};

template <typename T>
struct Array<T*>
{
    static void print()
    {
        std::cout << "T* sizoef: "<< sizeof(T*) << "\n";
    }
};

template <typename T1, typename T2, typename T3>
void foo(char a1[7], char a2[], char (&a3)[1], char (&a4)[], T1 x1, T2& x2, T3&& x3)
{
        Array<decltype(a1)>::print();
        Array<decltype(a2)>::print();
        Array<decltype(a3)>::print();
        Array<decltype(a4)>::print();
        Array<decltype(x1)>::print();
        Array<decltype(x2)>::print();
        Array<decltype(x3)>::print();
}

template <int T>
void Fun2(char buf1[T])
{
    std::cout << "sizeof buf[10]: " << sizeof(buf1) << "\n";
}

char x[1];

int main()
{
    char buf[1];
    //Fun1(buf, buf, buf);
    //Fun2(buf);
    extern char x[];
    foo(buf, buf, buf, x, x, x, x);
    return 0;
}

