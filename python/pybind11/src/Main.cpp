#include <string>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

#include <pybind11/embed.h>


#if defined(WIN32)
#include <signal.h>
#endif

namespace py = pybind11;
using namespace py::literals;

int g_reload = 0;
void SignalCapture(int no) {
    if (no == SIGINT) {
        std::cout << "SIGINT " << no << "\n";
        g_reload = 1; 
    }
}


/*
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
PYBIND11_EMBEDDED_MODULE(cpp, m) 
{
py::class_<Pet>(m, "Pet")
    .def(py::init<const std::string&>())
    .def("SetName", &Pet::SetName)
    .def("GetName", &Pet::GetName)
    ;
}
*/


void Test1()
{
    py::print("Hello, World!");
    py::exec(R"(
        l = [1,2,3,45, 'aaa', 'bbb']
        print("abc {} {} {}".format(111,9999, l))
    )");
    
    py::module sys = py::module::import("sys");
    py::print(sys.attr("path"));
}

struct Stack
{
    void Push(py::object obj) {
        m_vec.push_back(obj);
    }

    void Pop() { 
        if (m_vec.empty()) {
            return;
        }
        m_vec.pop_back();
    }

    void Set(int index, py::object obj) {
        if (!CheckIndex(index)) {
            return;
        }
        std::swap(m_vec[index], obj);
    }

    py::object Get(int index) {
        if (!CheckIndex(index)) {
            return py::none();
        }
        return m_vec[index];
    }

    void Swap(int a, int b) {
        if (a == b) {
            return;
        }
        if (!CheckIndex(a) || !CheckIndex(b)) {
            return;
        }
        std::swap(m_vec[a], m_vec[b]);
    }

    int Size() const {
        return (int)m_vec.size();
    }

    bool CheckIndex(int index) const {
        return 0 <= index && index < static_cast<int>(m_vec.size());
    }

    std::vector<py::object> m_vec;
};

Stack* g_stack = nullptr;
Stack* GetStack(int hdl)
{
    return g_stack;
}

PYBIND11_EMBEDDED_MODULE(Cpp, m) 
{
    py::class_<Stack>(m, "Stack")
        //.def(py::init<const std::string&>())
        .def("Push", &Stack::Push)
        .def("Pop", &Stack::Pop)
        .def("Set", &Stack::Set)
        .def("Get", &Stack::Get)
        .def("Size", &Stack::Size)
        ;

    m.def("GetStack", &GetStack, py::return_value_policy::reference);
}

void TestReload()
{
    if (1) {
        py::module b;
        if (b) {
            std::cout << "b is true\n";
        } else {
            std::cout << "b is true\n";
        }
        return ;
    }

    try {
        py::module a = py::module::import("a");
        py::function py_Add = a.attr("Add");
        py::function py_Length = a.attr("Length");

        int n = 0;
        std::string name;
        while (1) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            py::object result = py_Add(1,2);
            n = result.cast<int>();

            py::object s = a.attr("name");
            name = s.cast<std::string>();

            int len = py_Length("12").cast<int>();
            //std::cout << "name: " << name << " Add: " << n << " len: " << len << "\n";
            std::cout << " Add: " << n  << "\n";
            if (g_reload == 1) {
                g_reload = 0;
                std::cout << "reload script \n";
                a.reload();
                py_Add = py::function(a.attr("Add"));
                py_Length = py::function(a.attr("Length"));
            }
        }
    } catch (const std::exception & e) {
        std::cout << "exception: " << e.what() << "\n";
    }
}

void TestStack()
{
    g_stack = new Stack();
    try {
        py::module ts = py::module::import("ts");
        py::function py_oncall = ts.attr("OnCall");

        while (1) {
            std::this_thread::sleep_for(std::chrono::seconds(2));

            int index = 1;
            py::object result = py_oncall(index);
            int n = result.cast<int>();
            std::cout << "Cpp: Stack size: " << GetStack(index)->Size() << "\n";
        }
    } catch (const std::exception & e) {
        std::cout << "exception: " << e.what() << "\n";
    }
    delete g_stack;
}

int main()
{
    signal(SIGINT, SignalCapture);

    py::scoped_interpreter guard{};

    //TestReload();
    TestStack();

    system("pause");
    return 0;
}

