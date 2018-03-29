#include <iostream>
#include <vector>
#include <cassert>

#include "Cards.h"

using namespace gp_alg;

void test_autoSelect(std::string str_src, std::string str_hand)
{
    auto all_cards = initCard();
    auto src = utility::pickCardsFromString(str_src, &all_cards);
    auto hand = utility::pickCardsFromString(str_hand, &all_cards);

    std::cout << utility::printCardValue(src) << "\n";
    std::cout << utility::printCardValue(hand) << "\n";

    std::vector<Card> out;
    out.resize(hand.size());

    auto ret = autoSelect(src.data(), (int32_t)src.size(), hand.data(), (int32_t)hand.size(), out.data());
    if (ret) { 
        out = removeCardNull(out);
        std::cout << "success:" << utility::printCardValue(out) << "\n\n";
    } else {
        std::cout << "failed:" << __LINE__ << ":" << __FUNCTION__ << "\n\n";
    }
}

void test_parseCardType(std::string str_src)
{
    auto all_cards = initCard();
    auto src = utility::pickCardsFromString(str_src, &all_cards);
    std::cout << utility::printCardValue(hand) << "\n";

    auto ret = parseCardType(*src);
    if (ret.m_type) { 
        std::cout << "success: type:" << ret.m_type << "\n"
            << "\t\t value:" << ret.m_value << "\n";
            << "\t\t len:" << ret.m_type_len << "\n";
            << "\t\t"<< utility::printCardValue(src) << "\n\n";
    } else {
        std::cout << "failed:" << __LINE__ << ":" << __FUNCTION__ << "\n\n";
    }
}

int main()
{
    //411
    //test_autoSelect("A A 4 4 4 4",  "RJoker BJoker 7 7 7 7 3 3");
    //test_autoSelect("A A 4 4 4 4",  "RJoker BJoker 7 7 7 7 3 3");
    //test_autoSelect("A A 4 4 4 4",  "RJoker K K 7 7 7 7");

    //422
    //test_autoSelect("A A 4 4 4 4 3 3",  "RJoker K K K 7 7 7 7 5 5");
    //test_autoSelect("A A 4 4 4 4 3 3",  "RJoker K K K K 7 7 7 7 5 5");
    //test_autoSelect("A A 4 4 4 4 3 3",  "RJoker K K K K Q Q 7 7 7 7 5 5");
    //test_autoSelect("A A 4 4 4 4 3 3",  "RJoker K K K K Q Q Q 7 7 7 7 5 5");
    //test_autoSelect("A A 4 4 4 4 3 3",  "RJoker K K K K Q Q Q 7 7 7 7 6 6 6 5 5 5");

    return 0;
}