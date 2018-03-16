#include <iostream>
#include <memory>
#include <string>

class A
{
    A(const std::string& s)
        : m_str(s)
    {

    }
public:
    ~A()
    {

    }

    static std::shared_ptr<A> initA()
    {
        return std::make_shared<A>("hahah");
    }

    const std::string& getStr() const
    {
        return m_str;
    }

private:
    std::string m_str;
};

int main()
{
    auto a = A::initA();
    std::cout << a->getStr() << "\n";
    return 0;
}
