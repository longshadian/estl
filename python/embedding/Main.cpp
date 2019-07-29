
#define PY_SSIZE_T_CLEAN
#include <python3.7/Python.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>

std::string strErrorMsg;

/*
void log_python_exception()
{
    if (!::Py_IsInitialized()) {
        strErrorMsg = "Python 运行环境没有初始化！";
        return;
    }
    if (::PyErr_Occurred() != NULL) {
        PyObject *type_obj, *value_obj, *traceback_obj;
        ::PyErr_Fetch(&type_obj, &value_obj, &traceback_obj);
        if (value_obj == NULL)
            return;
        ::PyErr_NormalizeException(&type_obj, &value_obj, 0);
        if (PyString_Check(PyObject_Str(value_obj))) {
            strErrorMsg = PyBytes_AS_STRING(PyObject_Str(value_obj));
        }
    
        if (traceback_obj != NULL) {
            strErrorMsg += "Traceback:";
        
            PyObject * pModuleName = PyString_FromString("traceback");
            PyObject * pTraceModule = PyImport_Import(pModuleName);
            Py_XDECREF(pModuleName);
            if (pTraceModule != NULL) {
                PyObject * pModuleDict = PyModule_GetDict(pTraceModule);
                if (pModuleDict != NULL) {
                    PyObject * pFunc = PyDict_GetItemString(pModuleDict, "format_exception");
                    if (pFunc != NULL) {
                        PyObject * errList = PyObject_CallFunctionObjArgs(pFunc, type_obj, value_obj, traceback_obj, NULL);
                        if (errList != NULL) {
                            int listSize = PyList_Size(errList);
                            for (int i = 0; i < listSize; ++i)
                            {
                                strErrorMsg += PyString_AsString(PyList_GetItem(errList, i));
                            }
                        }
                    }
                }
                Py_XDECREF(pTraceModule);
            }
        }
        Py_XDECREF(type_obj);
        Py_XDECREF(value_obj);
        Py_XDECREF(traceback_obj);
    }
}
*/

int Test1()
{
    const char* fname = "./python/TestException.py";
    std::FILE* f = std::fopen(fname, "rb");
    if (!f) {
        printf("open file failed. [%s] \n", fname);
        return 0;
    }
    int ret = ::PyRun_SimpleFile(f, "test.py");

    printf("PyRun result: [%d] \n", ret);
    return 0;
}

int main()
{
    if (1) {
        const wchar_t* pypath = L"D:\\vspro\\pythontest\\Test\\Test\\python";
        std::wstring ws = ::Py_GetPath();
        ws += L";";
        ws += pypath;
        ::Py_SetPath(ws.c_str());
        //int v = ::PyRun_SimpleString("sys.path.append('D:\vspro\pythontest\Test\Test\python')");

        ::Py_Initialize();
        //int v = ::PyRun_SimpleString("sys.path.append('D:\vspro\pythontest\Test\Test\python')");
        //printf("py append: [%d] \n", v);

        Test1();
        ::Py_Finalize();
    }

    if (0) {
        printf("xxxxx sizeof(wchar_t) %llu\n", sizeof(wchar_t));
        printf("xxxxx sizeof(char16_t) %llu\n", sizeof(char16_t));
        printf("xxxxx sizeof(char32_t) %llu\n", sizeof(char32_t));
    }

    std::system("pause");
    return 0;
}

