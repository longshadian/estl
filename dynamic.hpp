#ifndef ESTL_DYNAMIC_H
#define ESTL_DYNAMIC_H

#include <iostream>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace estl
{

#define ESTL_DYNAMIC_APPLY(type, apply)         \
do {                                           \
    switch ((type)) {                          \
    case NULLT : apply(void*); break;        \
    case DOUBLE: apply(double); break;         \
    case UINT : apply(unsigned); break;        \
    case ARRAY : apply(Array);  break;         \
    case OBJECT : apply(Map); break;            \
    case BOOL : apply(bool); break;             \
    case STRING : apply(std::string); break;    \
    case default : apply(bool); abort();        \
    }                                           \
} while (0)

class Dynamic
{
public:
    enum class Type : char
    {
        NULLT,
        DOUBLE,
        UINT,
        INT,
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

    Dynamic(int v)
        : mType(Type::INT)
    {
        mData.mInt = v;
    }

    Dynamic(unsigned v)
        : mType(Type::UINT)
    {
        mData.mUInt = v;
    }

    Dynamic(double v)
        : mType(Type::DOUBLE)
    {
        mData.mDouble = v;
        std::cout << v << std::endl;
    }
    
    Dynamic(std::string name)
        : mType(Type::STRING)
    {
        mData.mString = name;
    }

   Dynamic(const char* name) 
       : mType(Type::STRING)
   {
       mData.mString = name;
   }

    ~Dynamic()
    {
        if (mType == Type::STRING)
            mData.mString.~basic_string();
    }

    Dynamic(const Dynamic& obj)
        : mType(Type::OBJECT)
    {

    }

    Dynamic& operator= (const Dynamic& obj)
    {
        mType = obj.mType;
        std::memcpy(&mData, &obj.mData, sizeof(Data));
        return *this;
    }

    Dynamic& operator[] (const Dynamic& obj)
    {
        return mData.mMap[obj];
    }

    bool operator< (const Dynamic& obj) const
    {
        if (mType != obj.mType)
            return mType < obj.mType;
#define ESTL_X(T)   return false;
        ESTL_X(T)
#undef ESTL_X
    }

    bool as_bool() const { return mData.mBool; }
    int as_int() const { return mData.mInt; }
    unsigned as_uint() const { return mData.mUInt; }
    double as_double() const { return mData.mDouble; }
    std::string as_string() const { return mData.mString; }
private:
    using Array = std::vector<Dynamic>;
    using Map = std::map<Dynamic, Dynamic>;
private:
    Type    mType;
    union Data
    {
        Data() {}
        ~Data() {}

        bool    mBool;
        unsigned mUInt;
        int     mInt;
        double  mDouble;
        std::string mString;
        Map mMap;
        Array mArray;
    } mData;

};

//////////////////////////////////////////////////////////////////////////
}

#endif // ESTL_DYNAMIC   
