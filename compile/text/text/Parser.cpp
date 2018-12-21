#include "Parser.h"
#include "Lexer.h"

idParser::idParser()
{
}

idParser::~idParser()
{
}

bool idParser::Start()
{
    while (true) {
        if (!GetNextToken()) {
            CommonError("get token error");
            return false;
        }
        if (Token()->AsStringView() == "#") {

        }
    }
}

void idParser::CommonError(std::string_view str)
{

}

bool idParser::GetNextToken()
{
    if (!m_lexer->GetToken(&m_token))
        return false;
    return true;
}

const idToken* idParser::Token() const
{
    return &m_token;
}

bool idParser::ParseGenerateSentinel()
{
    GetNextToken();
    if (Token()->AsStringView() != "\"") {
        return false;
    }

    GetNextToken();
    return true;
}
