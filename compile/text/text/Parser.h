#pragma once

#include <string>
#include <vector>

#include "Token.h" 

class idLexer;

class idParser
{
public:
					idParser();
                    ~idParser();
					idParser(const idParser&) = delete;
                    idParser& operator=(const idParser&) = delete;
                    idParser(idParser&&) = delete;
                    idParser& operator=(idParser&&) = delete;

    bool            Start();

private:
    void            CommonError(std::string_view str);
    bool            GetNextToken();
    const idToken*  Token() const;

    // 解析UHT哨兵
    bool            ParseGenerateSentinel();

private:
    idLexer*        m_lexer;
    idToken         m_token;
    bool            m_indent;
};

