#include "Cards.h"

#include <algorithm>
#include <sstream>
#include <cstring>
#include <array>

namespace gp_alg {


CardType::CardType() 
    : m_type(Type::Type_Null), m_value(Value::Card_Null), m_type_len(0)
{}

CardType::CardType(Type type, Value value, int32_t type_len) 
    : m_type(type), m_value(value), m_type_len(type_len)
{}

bool CardType::operator<(const CardType& rhs) const
{
    return detail::compareCardType(*this, rhs) == -1;
}

bool CardType::operator==(const CardType& rhs) const
{
    return detail::compareCardType(*this, rhs) == 0;
}

bool CardType::operator!=(const CardType& rhs) const
{
    return !(*this == rhs);
}


std::vector<Card> initCard()
{
    std::vector<Card> all_cards{};
    initCard(all_cards);
    std::sort(all_cards.begin(), all_cards.end(), PreGreaterSort());
    return all_cards;
}

//0-12	方块3~10-J-Q-K-A-2	13-25 梅花3~10-J-Q-K-A-2
//26-38 红桃3~10-J-Q-K-A-2	39-51 黑桃3~10-J-Q-K-A-2
//52 53	小王 大王
void initCard(std::vector<Card>& card)
{
    Card c{};
    for (int32_t i = 0; i < 52; ++i) {
        c = static_cast<Card>(makeColor(c, static_cast<Color>(i/13)));	    // 花色从数值0(方块)开始到数值3(黑桃)结束
        c = static_cast<Card>(makeValue(c, static_cast<Value>(i%13+1)));	// 牌值从数值1(牌值3)开始到数值13(牌值2)结束
        card.push_back(c);
    }
    c = static_cast<Card>(makeColor(c, Color::Diamond));    // 小王花色的数值为0
    c = static_cast<Card>(makeValue(c, Value::Card_B));    // 小王牌值的数值为14
    card.push_back(c);

    c = static_cast<Card>(makeColor(c, Color::Diamond));    // 大王花色的数值为0
    c = static_cast<Card>(makeValue(c, Value::Card_R));    // 大王牌值的数值为15
    card.push_back(c);
    //std::random_shuffle(card.begin(), card.end());
}

bool check(const Card* src, int32_t src_len,
           const Card* hand, int32_t hand_len)
{
    if (!hand || hand_len <= 0)
        return false;
    std::vector<Card> hand_out;
    hand_out.reserve(hand_len);
    CardType hand_type = parseCardType(hand, hand_len, &hand_out);

    // 手牌没有牌型
    if (hand_type.m_type == Type::Type_Null)
        return false;

    std::vector<Card> src_out;
    src_out.reserve(src_len);
    CardType src_type = parseCardType(src, src_len, &src_out);
    if (src_type.m_type == Type::Type_Null) {
        return false;
    }
    return src_type < hand_type;
}

bool playCard(const Card* src, int32_t src_len
    , const Card* selected,  int32_t selected_len
    , Card* out)
{
    if (!selected || selected_len <= 0)
        return false;

    //判断手牌是否合法
    std::vector<Card> selected_out;
    selected_out.reserve(selected_len);
    auto selected_type = parseCardType(selected, selected_len, &selected_out);
    if (selected_type.m_type == Type::Type_Null)
        return false;

    //目标牌是空的,手牌可以出
    if (!src || src_len == 0) {
        //不需要排序，hand已经是排序过的 
        for (int32_t i = 0; i < selected_len; ++i)
            out[i] = selected_out[i];
        return true;
    }

    //判断目标牌型
    std::vector<Card> src_out;
    src_out.reserve(src_len);
    CardType src_type = parseCardType(src, src_len, &src_out);
    if (src_type.m_type == Type::Type_Null) {
        return false;
    }

    if (src_type < selected_type) {
        for (int32_t i = 0; i < selected_len; ++i)
            out[i] = selected_out[i];
        return true;
    }
    return false;
}

CardType parseCardType(const Card* src, int32_t src_len, std::vector<Card>* out)
{
    // 超过16张牌不能组成合法牌型
    if (src_len > 16)
        return {};
    out->resize(src_len);

    if (src_len == 1) {
    // 单牌
        out->push_back(src[0]);
        return {Type::Type_Signle, getCardValue(src[0]), 1};
    }

    if (src_len == 2) {
    // 对子
        if (getCardValue(src[0]) == getCardValue(src[1])) {
            out->push_back(src[0]); out->push_back(src[1]);
            detail::sortCard(out);
            return {Type::Type_Double, getCardValue(src[0]), 1};
        }
        return {};
    }

    if (src_len == 3) {
        // 三张
        Value val = getCardValue(src[0]);
        if (val == getCardValue(src[1]) && val == getCardValue(src[2])) {
            out->push_back(src[0]); out->push_back(src[1]); out->push_back(src[2]);
            detail::sortCard(out);
            return {Type::Type_Triple, val, 1};
        }
        return {};
    }

    detail::SlotClassify classify{};
    if (!detail::createSlotClassify(src, src_len, &classify)) {
        return {};
    }
    // 长度最长的牌前面
    classify.sortByLengthDesc();

    if (src_len == 4) {
        // 炸弹
        if (classify.m_total_num_length == 4) {
            out->push_back(src[0]); out->push_back(src[1]); out->push_back(src[2]); out->push_back(src[3]);
            detail::sortCard(out);
            return {Type::Type_Bomb, getCardValue(src[0]), 1};
        }
        // AAA + ?
        if (classify.m_total_num_length == 3 && classify.m_slots[0].m_value == Value::Card_A) {
            detail::normalizeTypeList(classify, out);
            return CardType(Type::Type_31, classify.m_slots[0].m_value, 1);
        }
        return {};
    }

    std::vector<Card> src_sort(src, src + src_len);
    detail::sortCard(&src_sort);

    // 判断是否单顺,相同牌只有1张,牌值是顺子 9 8 10 J Q K A
    if (classify.m_sort_result[0].m_total_num == 1 && src_len <= 13 
        && detail::isSeries(src_sort.data(), src_len)) {
        detail::normalizeTypeList(classify, out);
        return CardType(Type::Type_Single_Ser, classify.m_sort_result[0].m_value, src_len);
    }

    // 判断是否双顺,相同牌有2张，牌值是顺子 JJ QQ KK AA
    if (classify.m_sort_result[0].m_total_num == 2 && src_len % 2 == 0 
        && detail::isSeries(classify, 2) == classify.m_total_num_length) {
        detail::normalizeTypeList(classify, out);
        return CardType(Type::Type_Double_Ser, classify.m_sort_result[0].m_value
            , classify.m_total_num_length);
    }

    // 判断是否三顺 JJJ QQQ KKK AAA
    if (classify.m_sort_result[0].m_total_num == 3 && src_len % 3 == 0 
        && detail::isSeries(classify, 3) == classify.m_total_num_length) {
        detail::normalizeTypeList(classify, out);
        return CardType(Type::Type_Triple_Ser, classify.m_sort_result[0].m_value
            , classify.m_total_num_length);
    }

    // 4+1 炸弹带单张
    if (src_len == 5 && classify.m_sort_result[0].m_total_num == 4) {
        detail::normalizeTypeList(classify, out);
        return CardType(Type::Type_41, classify.m_sort_result[0].m_value
            , classify.m_total_num_length);
    }

    return {};
}

CardType parseCardType(const Card* src, int32_t len)
{
    std::vector<Card> buffer;
    buffer.reserve(len);
    return parseCardType(src, len, &buffer);
}

CardType parseCardType(std::vector<Card>* src)
{
    std::vector<Card> buffer;
    buffer.reserve(src->size());
    auto type = parseCardType(src->data(), (int32_t)src->size(), &buffer);
    *src = std::move(buffer);
    return type;
}
std::vector<Card> removeCardNull(const Card* src, int32_t len)
{
    std::vector<Card> out;
    out.reserve(len);
    for (int32_t i = 0; i != len; ++i) {
        if (src[i] != 0) {
            out.push_back(src[i]);
        }
    }
    return out;
}

std::vector<Card> removeCardNull(const std::vector<Card>& src)
{
    return removeCardNull(src.data(), (int32_t)src.size());
}

/************************************************************************
 * 工具函数                                                                
 ************************************************************************/
namespace utility {

const CardNumString g_card_num_string[] =
{
    {Value::Card_Null,  "?"},
    {Value::Card_3,     "3"},
    {Value::Card_4,     "4"},
    {Value::Card_5,     "5"},
    {Value::Card_6,     "6"},
    {Value::Card_7,     "7"},
    {Value::Card_8,     "8"},
    {Value::Card_9,     "9"},
    {Value::Card_10,    "T"},
    {Value::Card_J,     "J"},
    {Value::Card_Q,     "Q"},
    {Value::Card_K,     "K"},
    {Value::Card_A,     "A"},
    {Value::Card_2,     "2"},
    {Value::Card_B,     "B"},
    {Value::Card_R,     "R"}
};

} // utility



namespace detail {

SlotClassify::SlotClassify()
    : m_slots()
    , m_sort_result()
    , m_total_num_length()
{
    int32_t card_3_val = static_cast<int32_t>(Value::Card_3);
    for (int32_t i = 0; i != 15; ++i) {
        Value v = static_cast<Value>(i + card_3_val);
        m_slots[i].m_value = v;
    }
}

void SlotClassify::sortByLengthDesc()
{
    std::vector<Slot> temp{};
    for (const auto& s : m_slots) {
        if (s.m_total_num != 0)
            temp.push_back(s);
    }
    std::sort(temp.begin(), temp.end()
        , [](const Slot& s1, const Slot& s2)
        {
            if (s1.m_total_num > s2.m_total_num) {
                return true;
            } else if (s1.m_total_num < s2.m_total_num) {
                return false;
            }
            return s1.m_value < s2.m_value;
        });

    m_sort_result = std::move(temp);
    if (!m_sort_result.empty()) {
        auto max_num = m_sort_result[0].m_total_num;
        m_total_num_length = std::count_if(m_sort_result.begin(), m_sort_result.end()
            , [max_num](const Slot& s)
            {
                return s.m_total_num == max_num;
            });
    }
}

void sortCard(std::vector<Card>* src)
{
    std::sort(src->begin(), src->end(), PreGreaterSort());
}

/*
int32_t getSameValueSeq(const Card* src, int32_t len, SameSeq* same_seq)
{
    if (len <= 0)
        return 0;

    Card cur_same = 1;
    Value last_same_val = getCardValue(src[0]);
    int32_t same_card_cnt = 0;
    for (int32_t i = 1; i < len; ++i) {
        if (getCardValue(src[i]) == last_same_val) {
            ++cur_same;
        } else {
            SameSeq& seq = same_seq[same_card_cnt++];
            seq.m_same_cnt = cur_same;
            seq.m_value = last_same_val;

            cur_same = 1;
            last_same_val = getCardValue(src[i]);
        }
    }
    SameSeq& seq = same_seq[same_card_cnt++];
    seq.m_same_cnt = cur_same;
    seq.m_value = last_same_val;
    for (int32_t i = 1; i < same_card_cnt; ++i) {
        for (int32_t j = 0; j < same_card_cnt - i; ++j) {
            if ((same_seq[j] & 0xFF00) < (same_seq[j + 1] & 0xFF00) ||
                ((same_seq[j] & 0xFF00) == (same_seq[j + 1] & 0xFF00) &&
                (same_seq[j] & 0x0F) > (same_seq[j + 1] & 0x0F))) {
                std::swap(same_seq[j], same_seq[j + 1]);
            }
        }
    }
    return same_card_cnt;
}
*/

bool createSlotClassify(const Card* src, int32_t len, SlotClassify* classify)
{
    if (len <= 0)
        return false;
    for (int32_t i = 0; i != len; ++i) {
        int32_t value_pos = getCardValue(src[i]) - Value::Card_3;
        int32_t color_pos = getCardColor(src[i]) - Color::Diamond;
        if (!(Value::Card_3 <= value_pos && value_pos <= Value::Card_R)
            || !(Color::Diamond <= color_pos && color_pos <= Color::Spade)) {
            return false;
        }
        Slot& s = classify->m_slots[value_pos];
        s.m_num[color_pos]++;
        s.m_total_num++;
    }
    return true;
}

void normalizeTypeList(const SlotClassify& classify, std::vector<Card>* out)
{
    for (const Slot& s : classify.m_sort_result) {
        for (int32_t color = 0; color != 4; ++color) {
            Value v = s.m_value;
            for (int32_t i = 0; i != s.m_num[color]; ++i) {
                Card c{};
                c = makeValue(c, v);
                c = makeColor(c, static_cast<Color>(color));
                out->push_back(c);
            }
        }
    }
}

/// 判断是否顺子
bool isSeries(const Card* src, int32_t len)
{
    if (len < 5)
        return false;
    for (int32_t i = 1; i < len; ++i) {
        Value card_val = getCardValue(src[i - 1]);
        Value next_val = getCardValue(src[i]);
        if (card_val < Value::Card_3 || next_val > Value::Card_A || next_val - card_val != 1)
            return false;
    }
    return true;
}

/// 判断是否多张顺，返回顺的次数
int32_t isSeries(const SlotClassify& classify, int32_t cnt)
{
    if (classify.m_sort_result.size() <= 1)
        return 0;
    int32_t seriles_len = 1;
    for (size_t i = 1; i != classify.m_sort_result.size(); ++i) {
        Value card_val = classify.m_sort_result[i-1].m_value;
        Value next_val = classify.m_sort_result[i].m_value;
        uint8_t card_count = classify.m_sort_result[i-1].m_total_num;
        uint8_t next_count = classify.m_sort_result[i].m_total_num;
        if (Value::Card_3 <= card_val && next_val <= Value::Card_A
            && next_val - card_val == 1
            && card_count == cnt && next_count == cnt
            ) {
            seriles_len++;
        } else {
            break;
        }
    }
    return seriles_len;
}

int32_t compareCardType(const CardType& type1, const CardType& type2)
{
    // TODO AAA + 1
    // 炸弹
    if (type1.m_type == Type::Type_Bomb && type2.m_type == Type::Type_Bomb) {
        if (type1.m_value == type2.m_value)
            return 0;
        return type1.m_value < type2.m_value ? -1 : 1;
    }
    if (type1.m_type == Type::Type_Bomb) {
        return 1;
    }
    if (type2.m_type == Type::Type_Bomb) {
        return -1;
    }

    // 其他牌型
    if (type1.m_type == type2.m_type) {
        if (type1.m_type_len != type2.m_type_len)
            return -1;
        if (type1.m_value > type2.m_value)
            return 1;
        else if (type1.m_value < type2.m_value)
            return -1;
        return 0;
    }
    return -1;
}


} // detail

} // gp_alg
