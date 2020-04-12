#pragma once

class Client
{
public:
    Client() = default;
    ~Client() = default;
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;
    Client(Client&&) = delete;
    Client& operator=(Client&&) = delete;

private:
};
