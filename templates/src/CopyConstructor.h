#pragma once

#include <string>
#include <utility>
#include <iostream>

struct Person1_Base
{
    Person1_Base() = default;

    Person1_Base(Person1_Base& rhs)
    {
    }
};

struct Person1 : Person1_Base
{
    std::string name;
    Person1()
        : name()
    {
    }

    template <typename T>
    Person1(T& t)
        : name(t.name)
    {
        std::cout << "TMPL-CONSTR Person: " << name << "\n";
    }

    ~Person1()
    {
    }
};

inline void TestCopyConstructor()
{
    Person1 p1;
    const Person1& rp1 = p1;
    //Person1 p3(p1); 
    Person1 p4 = p1;
    Person1 p5 = rp1;
}

