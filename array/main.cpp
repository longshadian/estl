#include <iostream>
#include <string>
#include <array>
#include <cstring>

#pragma pack(push, 1)
template <size_t L>
struct ObjString
{
    std::array<char, L> m_data;
    char m_null;

    const char* getString() const
    {
        return m_data.data();
    }

    bool setString(const std::string& s) 
    {
        if (s.size() > m_data.size()) {
            std::cout << "set string:" << s << " too len " << s.size() << " > " << m_data.size() << "\n";
            return false;
        }
        m_data.fill(0);
        std::memcpy(m_data.data(), s.data(), s.size());
        return true;
    }
};
#pragma pack(pop)


int main()
{
    ObjString<5> val;
    static_assert(std::is_pod<decltype(val)>::value, " is pod");
    std::memset(&val, 0, sizeof(val));
    std::cout << "sizeof:" << sizeof(val) << "\n";

    val.setString("a");
    std::cout << val.getString() << "|\n";

    val.setString("");
    std::cout << val.getString() << "|\n";

    return 0;
}
