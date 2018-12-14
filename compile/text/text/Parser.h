#pragma once

#include <string>
#include <vector>

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
private:
    idLexer*        m_lexer;
};

