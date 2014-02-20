#ifndef ESTL_DYNAMIC
#define ESTL_DYNAMIC

#include <map>
#include <string>
#include <vector>

namespace estl
{

class Dynamic
{
public:
    enum class Type : char
    {
        NULLT,
        DOUBLE,
        UINT,
        ARRAY,
        OBJECT,
        BOOL,
        STRING
    };
public:
    Dynamic()
    {
        new (&mData.mString) std::string();
    }

    ~Dynamic()
    {
        mData.mString.~basic_string();
    }

    std::
private:
    Type    mType;
    union Data
    {
        Data() {}
        ~Data() {}

        int     mInt;
        double  mDouble;
        std::string mString;
        std::map<Dynamic, Dynamic> mMap;
        std:vector<Dynamic> mArray;
    } mData;

};

//////////////////////////////////////////////////////////////////////////
}

#endif // ESTL_DYNAMIC   
