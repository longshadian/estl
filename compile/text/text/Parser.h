#pragma once

#include <string>
#include <vector>
#include <memory>

#include "Token.h" 
#include "Generation.h"

class idLexer;

enum class Indent
{
    Normal,
    Class,
    Struct,
};

class idParser
{
public:
					idParser(idLexer* m_lexer);
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
    void            ResetIndent();
    bool            IndentForClassStruct() const;

    // 解析UHT哨兵
    bool            ParseGenerateSentinel();
    bool            StartParse();
    bool            ParseUClass();
    bool            ParseUFunction();
    bool            ParseUProperty();

    std::shared_ptr<GenerateResult> ParseArgs();

private:
    idLexer*        m_lexer;
    idToken         m_token;
    Indent          m_indent;
    std::string     m_sentinel_name;
};

