#pragma once

#include <string>
#include <utility>
#include <iostream>

class Person
{
    std::string name;
public:

    template <typename T>
    Person(T&& t)
        : name(std::forward<T>(t))
    {
        std::cout << "TMPL-CONSTR Person: " << name << "\n";
    }

    ~Person()
    {
    }

    Person (Person& p) : name(p.name) {
        std::cout << "COPY-CONSTR reference Person: " << name << "\n";
    }

    Person (const Person& p) : name(p.name) {
        std::cout << "COPY-CONSTR Person: " << name << "\n";
    }

    Person (Person&& p) : name(std::move(p.name)) {
        std::cout << "MOVE-CONSTR Person: " << name << "\n";
    }
};

inline void TestPerson()
{
    std::string s1 = "str1";
    Person p1(s1);
    Person p2("temp");
    Person p3(p1); 
}

