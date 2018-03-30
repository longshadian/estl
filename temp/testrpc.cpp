
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <cstring>


struct Base
{
    Base() = default;
    virtual ~Base() = default;

    virtual std::vector<uint8_t> serialize() = 0;
    virtual void parse(std::vector<uint8_t> buffer) = 0;
};

struct Base_String : public Base
{
    Base_String() = default;
    virtual ~Base_String() = default;

    virtual std::vector<uint8_t> serialize() override
    {
        return std::vector<uint8_t>{m.begin(), m.end()};
    }

    virtual void parse(std::vector<uint8_t> buffer) override
    {
        m.assign(buffer.begin(), buffer.end());
    }

    std::string m;
};

struct Base_Int : public Base
{
    Base_Int() = default;
    virtual ~Base_Int() = default;

    virtual std::vector<uint8_t> serialize() override
    {
        std::vector<uint8_t> v;
        v.resize(sizeof(int32_t));
        std::memcpy(v.data(), &m, v.size());
        return v;
    }

    virtual void parse(std::vector<uint8_t> buffer) override
    {
        if (buffer.size() != sizeof(m))
            return;
        std::memcpy(&m, buffer.data(), buffer.size());
    }

    int32_t m;
};

template <typename T>
class Core
{
public:
    Core() {}
    ~Core() {}
    Core(const Core& rhs) = delete;
    Core& operator=(const Core& rhs) = delete;
    Core(Core&& rhs);
    Core& operator=(Core&& rhs);

    Core& OnSuccess(std::function<void(std::shared_ptr<T>)> success_cb)
    {
        m_cb = std::move(success_cb);
        return *this;
    }

    static void Parse(Base& b, std::vector<uint8_t> data)
    {
        b.parse(std::move(data));
    }

    void callback(std::vector<uint8_t> data)
    {
        auto t = std::make_shared<T>();
        Parse(*t, std::move(data));
        m_cb(std::move(t));
    }

    std::function<void(std::shared_ptr<T>)> m_cb;
};

template <typename T>
class ProtobufRpc
{
public:
    ProtobufRpc()
        : m_core(std::make_shared<Core<T>>())
    {
    }

    ~ProtobufRpc() = default;

    ProtobufRpc(const ProtobufRpc& rpc) = delete;
    ProtobufRpc& operator=(const ProtobufRpc& rpc) = delete;

    ProtobufRpc(ProtobufRpc&& rpc)
        : m_core(std::move(rpc.m_core))
    {
    }

    ProtobufRpc& operator=(ProtobufRpc&& rpc)
    {
        if (this != &rpc) {
            std::swap(m_core, rpc.m_core);
        }
        return *this;
    }

    ProtobufRpc&& OnSuccess(std::function<void(std::shared_ptr<T>)> cb)
    {
        m_core->OnSuccess(std::move(cb));
        return std::forward<ProtobufRpc>(*this);
    }

    std::function<void(std::vector<uint8_t>)> BindCB()
    {
        return std::bind(&Core<T>::callback, m_core, std::placeholders::_1);
    }

    std::shared_ptr<Core<T>> m_core;
};

template <typename T>
ProtobufRpc<T> makeRpc()
{
    return ProtobufRpc<T>{};
}

int main()
{
    auto t = std::make_shared<Base_Int>();
    t->m = 100000;
    std::function<void(std::vector<uint8_t>)> cb{};
    {
        auto pb_rpc = makeRpc<Base_Int>()
            .OnSuccess([](std::shared_ptr<Base_Int> pb)
            {
                std::cout << "int: " << pb->m << "\n";
            });

        cb = pb_rpc.BindCB();
    }
    cb(t->serialize());


    auto t1 = std::make_shared<Base_String>();
    t1->m = "123n;asoidfnpsdoij pqwoier isaufg";
    std::function<void(std::vector<uint8_t>)> cb_1{};
    {
        auto pb_rpc = makeRpc<Base_String>()
            .OnSuccess([](std::shared_ptr<Base_String> pb)
            {
                std::cout << "string: " << pb->m << "\n";
            });

        cb_1 = pb_rpc.BindCB();
    }
    cb_1(t1->serialize());

    return 0;
}
