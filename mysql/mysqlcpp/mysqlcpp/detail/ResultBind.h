#pragma once

#include <mysql.h>

#include <vector>
#include <memory>

namespace mysqlcpp {

class PreparedStatement;

namespace detail {


struct ResultBindBuffer
{
    ResultBindBuffer(size_t s, enum_field_types t) 
        : m_size(s)
        , m_type(t) 
        , m_buffer(nullptr)
        , buffer_()
    {
        if (s > 0) {
            buffer_.resize(s);
            m_buffer = buffer_.data();
        }
    }

    ~ResultBindBuffer()
    {

    }

    size_t              m_size;
    enum_field_types    m_type;
    char*               m_buffer;
private:
    std::vector<char>   buffer_;
};

class MYSQLCPP_EXPORT ResultBind
{
    struct FuckBool
    {
        bool m_bool;
    };

public:
	ResultBind();
	~ResultBind();

    ResultBind(const ResultBind& rhs) = delete;
    ResultBind& operator=(const ResultBind& rhs) = delete;

	bool bindResult(PreparedStatement& ps);
    const MYSQL_BIND* getMYSQL_BIND(size_t i) const;
    void clear();
private:
    std::vector<MYSQL_BIND>     m_bind;
    std::vector<FuckBool>       m_is_null;
    std::vector<FuckBool>       m_err;
    std::vector<unsigned long>  m_len;
    std::vector<std::shared_ptr<ResultBindBuffer>> m_bind_buffer;
};

} // detail

} // mysqlcpp
