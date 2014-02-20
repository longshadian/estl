#ifndef _ESTL_STRING_H_
#define _ESTL_STRING_H_

namespace estl 
{

template <typename C>
class string
{
public:
    typedef C value_type;
    string() {}
private:
    C*      pc_;
    size_t  n;
};

//////////////////////////////////////////////////////////////////////////
}

#endif
