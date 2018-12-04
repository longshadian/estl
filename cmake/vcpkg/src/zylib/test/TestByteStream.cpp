
#include "ByteStream.h"
#include "ByteStreamSerializer.h"

#include <string>
#include <iostream>

struct X
{
    bool        m_bool;
    int32_t     m_int32;
    int64_t     m_int64;
    std::string m_string;
    float       m_float;

    bool operator==(const X& rhs) const
    {
        return this->m_bool == rhs.m_bool 
            && this->m_int32 == rhs.m_int32 
            && this->m_int64 == rhs.m_int64
            && this->m_string == rhs.m_string
            && this->m_float == rhs.m_float
            ;
    }

    META(m_bool, m_int32, m_int64, m_string,m_float);

    void Write(ByteStream& bs) const
    {
        bs.Write(m_bool);
        bs.Write(m_int32);
        bs.Write(m_int64);
        bs.Write(m_string);
        bs.Write(m_float);
    }

    void Read(ByteStream& bs)
    {
        bs.Read(&m_bool);
        bs.Read(&m_int32);
        bs.Read(&m_int64);
        bs.Read(&m_string);
        bs.Read(&m_float);
    }
};

std::ostream& operator<<(std::ostream& ostm, const X& x) 
{
    ostm << "m_bool: " << x.m_bool << "\n"
        << "m_int32: " << x.m_int32 << "\n"
        << "m_int64: " << x.m_int64 << "\n"
        << "m_string: " << x.m_string << "\n"
        << "m_float: " << x.m_float << "\n"
        ;
    return ostm;
}

void TestByteStream()
{
    ByteStream bb{};
    X x{};
    x.m_bool = true;
    x.m_int32 = 123123132;
    x.m_int64 = int64_t(1312222222222222222);
    x.m_float = 123.4122f;
    x.m_string = "123bs0dofnabbbbbccccccef";

    try {
        x.Write(bb);
        X x2{};
        x2.Read(bb);
        std::cout << (int)(x == x2) << "\n";
        std::cout << x2 << "\n";
        x2.Read(bb);
    } catch (ByteStreamException e) {
        std::cout << e.what() << "\n";
    }
}

void TestByteStream_ByStream()
{
    ByteStream bb{};
    X x{};
    x.m_bool = true;
    x.m_int32 = 123123132;
    x.m_int64 = int64_t(1312222222222222222);
    x.m_float = 123.4122f;
    x.m_string = "123bs0dofnabbbbbccccccef";

    try {
        bb << x;

        std::cout << "size: " << bb.ReadSize() << "\n";

        X x2{};
        bb >> x2;
        std::cout << (int)(x == x2) << "\n";
        std::cout << x2 << "\n";
    } catch (ByteStreamException e) {
        std::cout << e.what() << "\n";
    }
}

int main()
{
    TestByteStream();
    TestByteStream_ByStream();

    system("pause");
    return 0;
}
