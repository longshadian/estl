#include <iostream>
#include <vector>
#include <cassert>
#include <sstream>

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
        removeCardNull(&out);
        std::cout << "success:" << utility::printCardValue(out) << "\n\n";
    } else {
        std::cout << "failed:" << __LINE__ << ":" << __FUNCTION__ << "\n\n";
    }
}

/*
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
*/

std::string strCardType(CardType card_type)
{
    std::ostringstream ostm{};
    ostm << utility::cardTypeToString(card_type.m_type) << " "
        << utility::cardValueToString(card_type.m_value) << " "
        << card_type.m_type_len;
    return ostm.str();
}

void testType()
{
    auto all_cards = initCard();
    auto src = utility::pickCardsFromString("2 A 4 3 5 6", &all_cards);

    auto card_type = parseCardType(&src);
    std::cout << strCardType(card_type) << "     " << utility::printCardValue(src) << "\n";
}

void testShunZi()
{
    // кЁвс
    //test_autoSelect("A 2 5 3 4 ",               "R B 2 3 4 5 6 7 8 9 T");
    //test_autoSelect("A 2 5 3 4 ",               "R B T 4 4 5 6 7 8 9 T");
    //test_autoSelect("6 7 2 5 3 4 ",             "R B T 4 4 5 6   9 T J Q K A 2");

    //test_autoSelect("3 3 4 4 ",                 "3 3 4 4 5 5 5 6 6");
    test_autoSelect("3 4 5 3 4 5 3 4 5",          "A A A J J J  Q Q Q K K K 7 7 7 8 8 8");
    //test_autoSelect("6 7 2 5 3 4 ",             "R B T 4 4 5 6 9 T J Q K A 2");

    /*
    test_autoSelect("A 2 A 4 4 4 4",  "R B 7 7 7 7 3 3");
    test_autoSelect("A A 4 4 4 4",  "R B 7 7 7 7 3 3");
    test_autoSelect("A A 4 4 4 4",  "R K K 7 7 7 7");
    */

    //422
    /*
    test_autoSelect("A A 4 4 4 4 3 3",  "R K K K 7 7 7 7 5 5");
    test_autoSelect("A A 4 4 4 4 3 3",  "R K K K K 7 7 7 7 5 5");
    test_autoSelect("A A 4 4 4 4 3 3",  "R K K K K Q Q 7 7 7 7 5 5");
    test_autoSelect("A A 4 4 4 4 3 3",  "R K K K K Q Q Q 7 7 7 7 5 5");
    test_autoSelect("A A 4 4 4 4 3 3",  "R K K K K Q Q Q 7 7 7 7 6 6 6 5 5 5");
    */
}

int main()
{
    //testType();
    testShunZi();

    system("pause");

    return 0;
}
