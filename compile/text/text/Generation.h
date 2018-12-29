#pragma 

#include <vector>
#include <string>

enum class GenerateType
{
    GTClass,
    GTFunction,
    GTProperty,
};

class GenerateParam
{
public:
    GenerateParam() = default;
    ~GenerateParam() = default;
    GenerateParam(const GenerateParam&) = default;
    GenerateParam& operator=(const GenerateParam&) = default;
    GenerateParam(GenerateParam&& rhs)
        : m_name(std::move(rhs.m_name))
        , m_value(std::move(rhs.m_value))
    {
    }

    GenerateParam& operator=(GenerateParam&& rhs)
    {
        if (this != &rhs) {
            std::swap(m_name, rhs.m_name);
            std::swap(m_value, rhs.m_value);
        }
        return *this;
    }

public:
    std::string m_name;
    std::string m_value;
};

class GenerateResult
{
public:
    GenerateResult()
        : m_type(GenerateType::GTClass)
        , m_params()
    {
    }

    ~GenerateResult() = default;
    GenerateResult(const GenerateResult&) = delete;
    GenerateResult& operator=(const GenerateResult&) = delete;
    GenerateResult(GenerateResult&&) = delete;
    GenerateResult& operator=(GenerateResult&&) = delete;

public:
    GenerateType                m_type;
    std::vector<GenerateParam>  m_params;
};

