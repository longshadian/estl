#ifndef _LUA_INTERFACE_H_
#define _LUA_INTERFACE_H_

#include <lua.hpp>
#include <stdarg.h>

#include <string>
#include <vector>
#include <sstream>

namespace luainterface {

std::string luaStackDump(lua_State* L);

template <typename T>
T luaToCppType(lua_State*L, int index);

template <>
int luaToCppType(lua_State*L, int index)
{
    return luaToIngeter(L, index);
}

inline
std::string luaToString(lua_State* L, int index)
{
    const char* p = lua_tostring(L, index);
    return p ? p : std::string();
}

inline 
double luaToNumber(lua_State* L, int index)
{
    return lua_tonumber(L, index);
}

inline 
int luaToIngeter(lua_State* L, int index)
{
    return lua_tointeger(L, index);
}

template <typename T>
T luaToUserdata(lua_State* L, int index)
{
    return static_cast<T>(lua_touserdata(L, index));
}

inline
bool luaToArray(lua_State* L, std::vector<int>& arr, int index = -1)
{
    if (lua_istable(L, index)) {
        int tn = lua_rawlen(L, index);
        for (int j = 1; j <= tn; ++j) {
            lua_rawgeti(L, index, j);
            int v = lua_tointeger(L, -1);
            arr.push_back(v);
            lua_pop(L, 1);
        }
        return true;
    }
    else {
        return false;
    }
}

template <typename T>
struct LuaFunParamTraits;

template <>
struct LuaFunParamTraits<int>
{
    static std::string paramType() { return "d"; }
};

template <>
struct LuaFunParamTraits<double>
{
    static std::string paramType() { return "f"; }
};

//不允许传递std::string，编译时会出错，因为变长参数无法传递非pod类型
template <>
struct LuaFunParamTraits<std::string>;

//const char* 是字符串类型
template <>
struct LuaFunParamTraits<const char*>
{
    static std::string paramType() { return "s"; }
};

//其余默认是指针类型
template <typename T>
struct LuaFunParamTraits<T*>
{
    static std::string paramType() { return "p"; }
};

template <typename T1>
std::string LuaFunArgv() { return LuaFunParamTraits<T1>::paramType(); }

template <typename T1, typename T2>
std::string LuaFunArgv() { return LuaFunParamTraits<T1>::paramType() + LuaFunParamTraits<T2>::paramType(); }

template <typename T1, typename T2, typename T3>
std::string LuaFunArgv() { return LuaFunParamTraits<T1>::paramType() + LuaFunParamTraits<T2>::paramType() + LuaFunParamTraits<T3>::paramType(); }

template <typename T1, typename T2, typename T3, typename T4>
std::string LuaFunArgv() { return LuaFunParamTraits<T1>::paramType() + LuaFunParamTraits<T2>::paramType() + LuaFunParamTraits<T3>::paramType() + LuaFunParamTraits<T4>::paramType(); }

inline
void luaSetField(lua_State* L, const char* key, int value)
{
    lua_pushinteger(L, value);
    lua_setfield(L, -2, key);
}

inline
void luaSetField(lua_State* L, const char* key, bool value)
{
    lua_pushboolean(L, value);
    lua_setfield(L, -2, key);
}

inline
void luaSetField(lua_State* L, const char* key, double value)
{
    lua_pushnumber(L, value);
    lua_setfield(L, -2, key);
}

inline
void luaSetField(lua_State* L, const char* key, const std::string& value)
{
    lua_pushstring(L, value.c_str());
    lua_setfield(L, -2, key);
}

class LuaState
{
public:
    LuaState() { L = luaL_newstate(); }
    ~LuaState() { lua_close(L); }

    void openlibs() { luaL_openlibs(L); }
    int registerCFUN(const char* module, const struct luaL_Reg* cppfun)
    {
        luaL_newlib(L, cppfun);
        lua_setglobal(L, module);
        return 1;
    }

    bool callLuaFun(const char* funname, int nret, const char* argv, ...);

    bool loadfile(const char* filename) 
    {
        return luaL_loadfile(L, filename) == 0 && lua_pcall(L, 0, 0, 0) == 0;
    }

    void pop(int n) { lua_pop(L, n); }
    void clear() { lua_settop(L, 0); }

    std::string luaToString(int index);
    int luaToInteger(int index);
    double luaToDouble(int index);
    
    std::string stackDump();
    lua_State* getLua_State() { return L; }
private:
    lua_State*  L;
private:
	LuaState(const LuaState&);
	LuaState& operator=(const LuaState&);
};

inline
bool LuaState::callLuaFun(const char* funname, int nret, const char* argv, ...)
{
    lua_getglobal(L, funname);
    int narg = 0;
    if (argv) {
        va_list vl;
        va_start(vl, argv);
        for (narg = 0; argv[narg]; narg++) {
            switch (argv[narg]) {
            case 'd' :
                lua_pushinteger(L, va_arg(vl,int));
                break;
            case 'f' :
                lua_pushnumber(L, va_arg(vl, double));
                break;
            case 's' :
                lua_pushstring(L, va_arg(vl, char*));
                break;
            case 'p' :
                lua_pushlightuserdata(L, va_arg(vl, void*));
                break;
            default:
                return false;
            }
        }
        va_end(vl);
    }
    return ::lua_pcall(L, narg, nret, 0) == 0;
}

inline
std::string LuaState::luaToString(int index)
{
    const char* p = lua_tostring(L, index);
    return p ? p : std::string();
}

inline
int LuaState::luaToInteger(int index)
{
    return lua_tointeger(L, index);
}

inline
double LuaState::luaToDouble(int index)
{
    return lua_tonumber(L, index);
}

inline
std::string luaStackDump(lua_State* L)
{
    std::ostringstream ostm;
    int top = lua_gettop(L);
    for (int i = 1; i <= top; ++i) {
        int t = lua_type(L, i);
        switch(t) {
        case LUA_TSTRING: {
            ostm << i << ":" << lua_tostring(L,i);
            break;
        }
        case LUA_TBOOLEAN: {
            ostm << i << ":" << (lua_toboolean(L,i) ? "true" : "false");
            break;
        }
        case LUA_TNUMBER: {
            ostm << i << ":" << lua_tonumber(L,i);
            break;
        }
        default: {
            ostm << i << ":" << lua_typename(L,t);
            break;
        }
        }
        ostm << " ";
    }
    return ostm.str();
}

inline
std::string LuaState::stackDump() 
{ 
    return luaStackDump(L);
}

bool executeScript(LuaState* ls, const char* funname)
{
    return ls->callLuaFun(funname, 0, NULL);
}

template <typename T1>
bool executeScript(LuaState* ls, const char* funname, T1 p1)
{
    std::string argv = luainterface::LuaFunArgv<T1>();
    return ls->callLuaFun(funname, 0, argv.c_str(), p1);
}

template <typename T1, typename T2>
bool executeScript(LuaState* ls, const char* funname, T1 p1, T2 p2)
{
    std::string argv = luainterface::LuaFunArgv<T1, T2>();
    return ls->callLuaFun(funname, 0, argv.c_str(), p1, p2);
}

template <typename T1, typename T2, typename T3>
bool executeScript(LuaState* ls, const char* funname, T1 p1, T2 p2, T3 p3)
{
    std::string argv = luainterface::LuaFunArgv<T1, T2, T3>();
    return ls->callLuaFun(funname, 0, argv.c_str(), p1, p2, p3);
}

template <typename T1, typename T2, typename T3, typename T4>
bool executeScript(LuaState* ls, const char* funname, T1 p1, T2 p2, T3 p3, T4 p4)
{
    std::string argv = luainterface::LuaFunArgv<T1, T2, T3, T4>();
    return ls->callLuaFun(funname, 0, argv.c_str(), p1, p2, p3, p4);
}

//////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////
}

#endif

