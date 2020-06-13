#ifndef _LUASTACK_H_
#define _LUASTACK_H_

#include <lua.hpp>
#include <stdarg.h>

#include <string>
#include <iostream>
#include <sstream>

namespace luainterface {

class LuaState;
std::string stackDump(lua_State* L);
std::string stackDump(LuaState* p);

class LuaState
{
public:
    LuaState() { L = ::luaL_newstate(); }
    LuaState() { L = ::luaL_newstate(); }
    ~LuaState() { ::lua_close(L); }

    void openlibs() { ::luaL_openlibs(L); }
    int registerCFUN(const char* module, const struct luaL_Reg* cppfun)
    {
        ::luaL_register(L, module, cppfun);
        return 1;
    }

    bool callLuaFun(const char* funname, int nret, const char* argv, ...);
    bool loadfile(const char* filename) 
    {
        return ::luaL_loadfile(L, filename) == 0 &&
             ::lua_pcall(L, 0, 0, 0) == 0;
    }

    void pop(int n) { ::lua_pop(L, n); }
    void clear() { ::lua_settop(L, 0); }
    
    lua_State* getLua_State() { return L; }
private:
    lua_State*  L;
private:
	LuaState(const LuaState&);
	LuaState& operator=(const LuaState&);
};

bool LuaState::callLuaFun(const char* funname, int nret, const char* argv, ...)
{
	//::luaL_checkstack(L, 128, "stack overflow");
    ::lua_getglobal(L, funname);
    //std::cout << stackDump(L) << std::endl;
    int narg;
    va_list vl;
    va_start(vl, argv);
    for (narg = 0; argv[narg]; narg++) {
        switch (argv[narg]) {
        case 'i' :
            ::lua_pushinteger(L, va_arg(vl,int));
            break;
        case 'n' :
            ::lua_pushnumber(L, va_arg(vl, double));
            break;
        case 's' :
            ::lua_pushstring(L, va_arg(vl, char*));
            break;
        default:
            return false;
        }
    }
    va_end(vl);

    return ::lua_pcall(L, narg, nret, 0) == 0;
}

std::string stackDump(lua_State* L)
{
    std::ostringstream ostm;
    int top = lua_gettop(L);
    for (int i = 1; i <= top; ++i) {
        int t = lua_type(L, i);
        switch(t) {
        case LUA_TSTRING: {
            ostm << lua_tostring(L,i);
            break;
        }
        case LUA_TBOOLEAN: {
            ostm << (lua_toboolean(L,i) ? "true" : "false");
            break;
        }
        case LUA_TNUMBER: {
            ostm << lua_tonumber(L,i);
            break;
        }
        default: {
            ostm << lua_typename(L,t);
            break;
        }
        }
        ostm << " ";
    }
    return ostm.str();
}

std::string stackDump(LuaState* p) { return stackDump(p->getLua_State()); }

//////////////////////////////////////////////////////////////////////////
}

#endif

