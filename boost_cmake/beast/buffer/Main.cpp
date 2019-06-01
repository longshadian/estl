#include <boost/beast.hpp>
#include <iostream>

template <typename B>
void PrintBuffer(const B& buffer)
{
    std::cout << "\n";
    std::cout << "capacity: " << buffer.capacity() << "\n";
    std::cout << "max_size: " << buffer.max_size() << "\n";
    if (buffer.size() != 0) {
        auto input = buffer.data();
        std::string content{};
        content.resize(input.size());
        auto* p = static_cast<const char*>(input.data());
        std::copy(p, p + input.size(), content.begin());
        std::cout << "size:     " << buffer.size() << "\t\t" << content << "\n";
    } else {
        std::cout << "size:     " << buffer.size() << "\n";
    }
}

std::string ReadN(boost::beast::flat_buffer* buffer, std::size_t n)
{
    if (buffer->size() < n)
        return {};
    auto input = buffer->data();
    std::string content{};
    content.resize(n);
    auto* p = static_cast<const char*>(input.data());
    std::copy(p, p + n, content.begin());
    buffer->consume(n);
    return content;
}

void TestBasicFlatBuffer()
{
}

void TestBasicMultiBuffer()
{

}

void TestFlatBuffer()
{
    try {
        boost::beast::flat_buffer buffer{};
        PrintBuffer(buffer);
        std::string s = "12";

        auto output = buffer.prepare(2);
        std::copy(s.cbegin(), s.cend(), static_cast<char*>(output.data()));
        buffer.commit(2);
        PrintBuffer(buffer);

        output = buffer.prepare(2);
        s = "ab";
        std::copy(s.cbegin(), s.cend(), static_cast<char*>(output.data()));
        buffer.commit(2);
        PrintBuffer(buffer);

        std::cout << "read: " << ReadN(&buffer, 3) << "\n";
        PrintBuffer(buffer);

        output = buffer.prepare(3);
        s = "xyz";
        std::copy(s.cbegin(), s.cend(), static_cast<char*>(output.data()));
        buffer.commit(3);
        PrintBuffer(buffer);

        //buffer.reserve(3);
    } catch (const std::exception& e) {
        std::cout << "TestFlatBuffer exception: " << e.what() << "\n";
    }
}

void TestFlatStaticBuffer()
{
}

void TestStaticBuffer()
{

}

void TestMultiBuffer()
{
}

int main()
{
    if (1) {
        TestFlatBuffer();
        std::cout << "********************************\n";
    }
    if (1) {
        TestMultiBuffer();
        std::cout << "********************************\n";
    }

    return 0;
}

