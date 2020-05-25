//**************************************************************************************************
/// imitate folly::dynamic
/// https://github.com/facebook/folly
//**************************************************************************************************

#ifndef ESTL_DYNAMIC_H
#define ESTL_DYNAMIC_H

#include <iostream>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace estl {

#define ESTL_DYNAMIC_APPLY(type, apply)         \
do {                                           \
    switch ((type)) {                          \
    case Dynamic::Type::NULLT : apply(void*); break;        \
    case Dynamic::Type::DOUBLE: apply(double); break;         \
    case Dynamic::Type::UINT : apply(unsigned); break;        \
    case Dynamic::Type::ARRAY : apply(Array);  break;         \
    case Dynamic::Type::OBJECT : apply(Object); break;            \
    case Dynamic::Type::BOOL : apply(bool); break;             \
    case Dynamic::Type::STRING : apply(std::string); break;    \
    default : apply(bool); abort();        \
    }                                           \
} while (0)

class Dynamic
{
public:
    enum Type
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
    Dynamic(Type t = Type::OBJECT) : m_type(t)
    {
        switch (m_type)
        {
        case estl::Dynamic::Type::NULLT:
        case estl::Dynamic::Type::DOUBLE:
        case estl::Dynamic::Type::UINT:
        case estl::Dynamic::Type::BOOL:
            break;
        case estl::Dynamic::Type::ARRAY:
            new (&m_data.m_array) Array();
            break;
        case estl::Dynamic::Type::OBJECT:
            new (&m_data.m_object) Object();
            break;
        case estl::Dynamic::Type::STRING:
            new (&m_data.m_string) std::string();
            break;
        default:
            break;
        }
    }
    Dynamic(int v) : m_type(Type::UINT) { m_data.m_uint = static_cast<unsigned>(v); }
	Dynamic(double v) : m_type(Type::DOUBLE) { m_data.m_double = v; }
    Dynamic(const std::string& name) : m_type(Type::STRING) 
    {
        new (&m_data.m_string) std::string(name);
    }
    Dynamic(const char* name) : m_type(Type::STRING)
    {
        new (&m_data.m_string) std::string(name);
    }

    ~Dynamic() { destroy(); }

	Dynamic(const Dynamic& obj)
	{
        m_type = obj.m_type;
#define ESTL_X(T) new (getAddress<T>()) T(*const_cast<Dynamic*>(&obj)->getAddress<T>())
		ESTL_DYNAMIC_APPLY(m_type, ESTL_X);
#undef ESTL_X
	}

    Dynamic& operator= (const Dynamic& obj)
    {
		if (this != &obj) {
			destroy();
            m_type = obj.m_type;
#define ESTL_X(T) new (getAddress<T>()) T(*const_cast<Dynamic*>(&obj)->getAddress<T>())
			ESTL_DYNAMIC_APPLY(m_type, ESTL_X);
#undef ESTL_X
		}
        return *this;
    }

    bool operator< (const Dynamic& obj) const
    {
        if (m_type != obj.m_type)
            return m_type < obj.m_type;
#define ESTL_X(T) return *const_cast<Dynamic*>(this)->getAddress<T>() < \
                         *const_cast<Dynamic*>(&obj)->getAddress<T>()
        ESTL_DYNAMIC_APPLY(m_type, ESTL_X);
#undef ESTL_X
    }

    Dynamic& operator[] (const Dynamic& obj)
    {
		if (!isArray() && !isObject())
			abort();
		if (isArray())
			return at(obj);
        return m_data.m_object[obj];
    }

    Dynamic& at(const Dynamic& idx)
    {
        if (!isArray() && !isObject())
            abort();
        if (isArray()) {
            if (!idx.isInt())
                abort();
            Array& arr = *getAddress<Array>();
            return arr[idx.asInt()];
        }
       auto it = m_data.m_object.find(idx);
       if (it == m_data.m_object.end())
           abort();
       return it->second;
    }
    const Dynamic& at(const Dynamic& idx) const
    {
        return const_cast<Dynamic*>(this)->at(idx);
    }

    void push_back(const Dynamic& obj)
    {
        if (!isArray())
            abort();
        m_data.m_array.push_back(obj);
    }

    size_t size() const
    {
        if (!isArray() && !isObject())
            abort();
        if (isArray())
            return m_data.m_array.size();
        return m_data.m_object.size();
    }

    bool empty() const
    {
        if (!isNull())
            return false;
        return !size();
    }

    bool asBool() const { return m_data.m_bool; }
    int asInt() const { return static_cast<int>(m_data.m_uint); }
    unsigned asUInt() const { return m_data.m_uint; }
    double asDouble() const { return m_data.m_double; }
    const std::string& asString() const { return m_data.m_string; }
	const std::map<Dynamic,Dynamic>& asObject() const { return m_data.m_object; }

	bool isBool() const { return m_type == Type::BOOL; }
	bool isNumeric() const { return m_type == Type::UINT || m_type == Type::DOUBLE; }
	bool isInt() const { return m_type == Type::UINT; }
	bool isDouble() const { return m_type == Type::DOUBLE; }
	bool isString() const { return m_type == Type::STRING; }
	bool isNull() const { return m_type == Type::NULLT; }
	bool isObject() const { return m_type == Type::OBJECT; }
    bool isArray() const { return m_type == Type::ARRAY; }
private:
	template<typename T> struct GetAddrImpl;
    using Array = std::vector<Dynamic>;
    using Object = std::map<Dynamic, Dynamic>;

	template<typename T>
	T* getAddress() 
	{
		return GetAddrImpl<T>::get(m_data);
	}

	void destroy()
	{
		switch (m_type) {
		case Type::NULLT : break;
		case Type::BOOL : m_data.m_bool = false; break;
		case Type::DOUBLE : m_data.m_double = 0.0; break;
		case Type::UINT : m_data.m_uint = 0; break;
		case Type::STRING : m_data.m_string.~basic_string(); break;
		case Type::OBJECT : m_data.m_object.~map(); break;
		case Type::ARRAY : m_data.m_array.~vector(); break;
		}
		m_type = Type::NULLT;
	}
private:
    Type    m_type;
    union Data
    {
        Data() {}
        ~Data() {}

		void*   m_nullt;
        bool    m_bool;
        unsigned m_uint;
        double  m_double;
        std::string m_string;
        Object m_object;
        Array m_array;
    } m_data;

};

template<typename T> struct Dynamic::GetAddrImpl { }; 

template<> struct Dynamic::GetAddrImpl<void*> 
{
	static void** get(Data& d) { return &d.m_nullt; }
};

template<> struct Dynamic::GetAddrImpl<bool>
{
	static bool* get(Data& d) { return &d.m_bool; }
};

template<> struct Dynamic::GetAddrImpl<unsigned>
{
	static unsigned* get(Data& d) { return &d.m_uint; }
};

template<> struct Dynamic::GetAddrImpl<double>
{
	static double* get(Data& d) { return &d.m_double; }
};

template<> struct Dynamic::GetAddrImpl<std::string>
{
	static std::string* get(Data& d) { return &d.m_string; }
};

template<> struct Dynamic::GetAddrImpl<Dynamic::Array>
{
	static Array* get(Data& d) { return &d.m_array; }
};

template<> struct Dynamic::GetAddrImpl<Dynamic::Object>
{
	static Object* get(Data& d) { return &d.m_object; }
};

//////////////////////////////////////////////////////////////////////////
}

#endif // ESTL_DYNAMIC   
