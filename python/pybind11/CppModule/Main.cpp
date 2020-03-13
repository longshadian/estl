#include <string>
#include <iostream>
#include <thread>
#include <chrono>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

struct Pet
{
    Pet(const std::string& name)
        : m_name(name)
    {

    }

    const std::string& GetName() const { return m_name; }
    void SetName(const std::string& name) { m_name = name; }

    std::string m_name;
};

int Subtract(int a, int b)
{
    return a - b;
}

Pet* GlobalPet()
{
    static Pet p("global pet");
    return &p;
}

PYBIND11_MODULE(CppModule, m) 
{
py::class_<Pet>(m, "Pet")
    .def(py::init<const std::string&>())
    .def("SetName", &Pet::SetName)
    .def("GetName", &Pet::GetName)
    ;

    m.def("Subtract", &Subtract);

    m.def("GlobalPet", &GlobalPet, py::return_value_policy::reference);
}



