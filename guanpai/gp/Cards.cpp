#include "Cards.h"

#include <algorithm>
#include <sstream>
#include <cstring>
#include <array>

namespace gp_alg {

Face getCardFace(Card card)
{
    Value v = getCardValue(card);
    switch (v)
    {
    case Value::Card_Null:  return Face::Face_Null;
    case Value::Card_3:     return Face::Face_3;
    case Value::Card_4:     return Face::Face_4;
    case Value::Card_5:     return Face::Face_5;
    case Value::Card_6:     return Face::Face_6;
    case Value::Card_7:     return Face::Face_7;
    case Value::Card_8:     return Face::Face_8;
    case Value::Card_9:     return Face::Face_9;
    case Value::Card_10:    return Face::Face_10;
    case Value::Card_J:     return Face::Face_J;
    case Value::Card_Q:     return Face::Face_Q;
    case Value::Card_K:     return Face::Face_K;
    case Value::Card_A:     return Face::Face_A;
    case Value::Card_2:     return Face::Face_2;
    case Value::Card_B:     return Face::Face_B;
    case Value::Card_R:     return Face::Face_R;
    default:
        break;
    }
    return Face::Face_Null;
}


CardType::CardType() 
    : m_type(Type::Type_Null), m_value(Value::Card_Null), m_type_len(0)
{
}

CardType::CardType(Type type, Value value, int32_t type_len) 
    : m_type(type), m_value(value), m_type_len(type_len)
{
}

CardType::~CardType()
{
}

CardType::CardType(const CardType& rhs)
    : m_type(rhs.m_type)
    , m_value(rhs.m_value)
    , m_type_len(rhs.m_type_len)
{
}

CardType& CardType::operator=(const CardType& rhs)
{
    if (this != &rhs) {
        this->m_type = rhs.m_type;
        this->m_value = rhs.m_value;
        this->m_type_len = rhs.m_type_len;
    }
    return *this;
}

CardType::CardType(CardType&& rhs)
    : m_type(std::move(rhs.m_type))
    , m_value(std::move(rhs.m_value))
    , m_type_len(std::move(rhs.m_type_len))
{
}

CardType& CardType::operator=(CardType&& rhs)
{
    if (this != &rhs) {
        std::swap(this->m_type, rhs.m_type);
        std::swap(this->m_value, rhs.m_value);
        std::swap(this->m_type_len, rhs.m_type_len);
    }
    return *this;
}

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

bool playCard(const Card* src, int32_t src_len
    , const Card* hand,  int32_t hand_len
    , Card* out, CardType* out_type)
{
    if (!hand || hand_len <= 0)
        return false;

    // 判断手牌是否合法
    std::vector<Card> hand_out;
    hand_out.reserve(hand_len);
    auto hand_type = parseCardType(hand, hand_len, &hand_out);
    if (hand_type.m_type == Type::Type_Null)
        return false;

    // 目标牌是空的,手牌可以出
    if (!src || src_len == 0) {
        std::copy(hand_out.begin(), hand_out.end(), out);
        if (out_type)
            *out_type = std::move(hand_type);
        return true;
    }

    // 判断目标牌型
    std::vector<Card> src_out;
    src_out.reserve(src_len);
    CardType src_type = parseCardType(src, src_len, &src_out);
    if (src_type.m_type == Type::Type_Null) {
        return false;
    }

    if (src_type < hand_type) {
        std::copy(hand_out.begin(), hand_out.end(), out);
        if (out_type)
            *out_type = std::move(hand_type);
        return true;
    }
    return false;
}

bool autoSelect(const Card* src, int32_t src_len
    , const Card* hand, int32_t hand_len
    , Card* out)
{
    if (!src || src_len == 0)
        return false;

    std::vector<Card> src_out{};
    src_out.reserve(src_len);
    auto src_type = parseCardType(src, src_len, &src_out);

    detail::SlotClassify classify{};
    if (!detail::createSlotClassify(src, src_len, &classify))
        return false;

    switch (src_type.m_type) {
    case Type::Type_Signle: {
        std::vector<Card> temp_out;
        temp_out.reserve(hand_len);
        if (detail::searchAtomic(&classify, src_out[0], 1, &temp_out)) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        break;
    } case Type::Type_Double: {
        std::vector<Card> temp_out;
        temp_out.reserve(hand_len);
        if (detail::searchAtomic(&classify, src_out[0], 2, &temp_out)) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        break;
    } case Type::Type_Triple: {
        std::vector<Card> temp_out;
        temp_out.reserve(hand_len);
        if (detail::searchAtomic(&classify, src_out[0], 3, &temp_out)) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        break;
    } case Type::Type_Single_Ser: {
        std::vector<Card> temp_out;
        temp_out.reserve(hand_len);
        int32_t src_count = src_type.m_type_len;
        if (detail::searchSeries(&classify, src_out[0], src_out[src_count - 1], 1, &temp_out)) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        break;
    }
    case Type::Type_Double_Ser: {
        std::vector<Card> temp_out(hand_len);
        int32_t src_count = src_type.m_type_len;
        if (detail::searchSeries(&classify, src_out[0], src_out[2 * (src_count - 1)], 2, &temp_out)) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        break;
    }
    case Type::Type_Triple_Ser: {
        std::vector<Card> temp_out(hand_len);
        int32_t src_count = src_type.m_type_len;
        if (detail::searchSeries(&classify, src_out[0], src_out[3 * (src_count - 1)], 3, &temp_out)) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        break;
    }
    case Type::Type_31: {
        break;
    }
    case Type::Type_Bomb: {
        std::vector<Card> temp_out{};
        temp_out.reserve(hand_len);
        if (detail::searchBomb(&classify, src_out[0], &temp_out)) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        break;
    }
    default:
        break;
    }

    if (src_type.m_type != Type::Type_31 && src_type.m_type != Type::Type_Bomb) {
        // 选择AAA炸弹，默认不带牌
        std::vector<Card> temp_out{};
        temp_out.reserve(hand_len);
        if (detail::searchBomb_AAA(&classify, src_out[0], &temp_out)) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
    }

    if (src_type.m_type != Type::Type_Bomb) {
        // 选择炸弹
        std::vector<Card> temp_out{};
        temp_out.reserve(hand_len);
        if (detail::searchBomb(&classify, src_out[0], &temp_out)) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
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
        return CardType(Type::Type_Bomb, classify.m_sort_result[0].m_value
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

void removeCardNull(const Card* src, int32_t len, std::vector<Card>* out)
{
    out->reserve(len);
    for (int32_t i = 0; i != len; ++i) {
        if (src[i] != Value::Card_Null) {
            out->push_back(src[i]);
        }
    }
}

void removeCardNull(std::vector<Card>* src)
{
    auto it = std::remove(src->begin(), src->end(), Value::Card_Null);
    src->erase(it, src->end());
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

Slot::Slot()
{
}

Slot::~Slot()
{
}

Card Slot::popOneCard()
{
    Card c{};
    if (m_total_num <= 0)
        return c;
    for (size_t i = 0; i != m_colors.size(); ++i) {
        if (m_colors[i] > 0) {
            m_colors[i]--;
            m_total_num--;
            c = makeCard(static_cast<Color>(i + Color::Diamond), m_value);
            break;
        }
    }
    return c;
}

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

SlotClassify::~SlotClassify()
{

}

SlotClassify::SlotClassify(const SlotClassify& rhs)
    : m_slots(rhs.m_slots)
    , m_sort_result(rhs.m_sort_result)
    , m_total_num_length(rhs.m_total_num_length)
{
}

SlotClassify& SlotClassify::operator=(const SlotClassify& rhs)
{
    if (this != &rhs) {
        this->m_slots = rhs.m_slots;
        this->m_sort_result = rhs.m_sort_result;
        this->m_total_num_length = rhs.m_total_num_length;
    }
    return *this;
}

SlotClassify::SlotClassify(SlotClassify&& rhs)
    : m_slots(std::move(rhs.m_slots))
    , m_sort_result(std::move(rhs.m_sort_result))
    , m_total_num_length(std::move(rhs.m_total_num_length))
{
}

SlotClassify& SlotClassify::operator=(SlotClassify&& rhs)
{
    if (this != &rhs) {
        std::swap(m_slots, rhs.m_slots);
        std::swap(m_sort_result, rhs.m_sort_result);
        std::swap(m_total_num_length, rhs.m_total_num_length);
    }
    return *this;
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

const Slot* SlotClassify::getSlotByValue(Value v) const
{
    if (!(Value::Card_3 <= v && v <= Value::Card_R))
        return nullptr;
    int32_t delta = v - Value::Card_3;
    return &m_slots[delta];
}

Slot* SlotClassify::getSlotByValue(Value v)
{
    if (!(Value::Card_3 <= v && v <= Value::Card_R))
        return nullptr;
    int32_t delta = v - Value::Card_3;
    return &m_slots[delta];
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
        Value val = getCardValue(src[i]);
        Slot* s = classify->getSlotByValue(val);
        if (!s)
            return false;
        int32_t color_pos = getCardColor(src[i]) - Color::Diamond;
        if (!(Color::Diamond <= color_pos && color_pos <= Color::Spade)) {
            return false;
        }
        s->m_colors[color_pos]++;
        s->m_total_num++;
    }
    return true;
}

void normalizeTypeList(const SlotClassify& classify, std::vector<Card>* out)
{
    for (const Slot& s : classify.m_sort_result) {
        for (int32_t color = 0; color != 4; ++color) {
            Value v = s.m_value;
            for (int32_t i = 0; i != s.m_colors[color]; ++i) {
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
    // 炸弹
    if (type1.m_type == Type::Type_Bomb && type2.m_type == Type::Type_Bomb) {
        if (type1.m_value == type2.m_value)
            return 0;
        return type1.m_value < type2.m_value ? -1 : 1;
    }

    // 炸弹
    if (type1.m_type == Type::Type_Bomb) {
        return 1;
    }
    if (type2.m_type == Type::Type_Bomb) {
        return -1;
    }

    // AAA + ?
    if (type1.m_type == Type::Type_31) {
        return 1;
    }
    if (type2.m_type == Type::Type_31) {
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

bool searchAtomic(SlotClassify* classify, Card src_min, int32_t count
    , std::vector<Card>* out)
{
    Value src_min_value = getCardValue(src_min);
    for (int32_t n = count; n < 4; ++n) {
        for (Value val = static_cast<Value>(src_min_value + 1)
            ; val <= Value::Card_R
            ; val = static_cast<Value>(val + 1)) {
            Slot* slot = classify->getSlotByValue(val);
            if (!slot)
                return false;
            if (slot->m_total_num == n) {
                while (count--) {
                    out->push_back(slot->popOneCard());
                }
                return true;
            }
        }
    }
    return false;
}

bool searchSeries(SlotClassify* classify
    , Card src_min
    , Card src_max
    , int32_t src_count
    , std::vector<Card>* out)
{
    int32_t src_len = searchSeries_CalcLen(src_min, src_max);
    Value src_min_val = getCardValue(src_min);
    if (src_min_val == Value::Card_A) {
        // 先选2开始的顺
        SlotClassify classifyBk = *classify;
        std::vector<Card> temp_out{};
        auto ret = searchSeries_From2(&classifyBk, src_len, src_count, &temp_out);
        if (ret) {
            *classify = std::move(classifyBk);
            *out = std::move(temp_out);
            return true;
        }
        // 选择从3开始
        return searchSeries_From3(classify, Value::Card_3, src_len, src_count, out);
    } else if (src_min_val == Value::Card_2) {
        // 选择从3开始
        return searchSeries_From3(classify, Value::Card_3, src_len, src_count, out);
    }

    // 其他情况下，选择src_min_val+1开始的顺
    return searchSeries_From3(classify, static_cast<Value>(src_min_val + 1), src_len, src_count, out);
}

// 计算顺子长度
int32_t searchSeries_CalcLen(Card src_min, Card src_max)
{
    Value src_min_value = getCardValue(src_min);
    Value src_max_value = getCardValue(src_max);

    if (src_min_value == Value::Card_A) {
        if (src_max_value == Value::Card_A) {
            return 1;
        } else if (src_max_value == Value::Card_2) {
            return 2;
        } else if (src_max_value >= Value::Card_3) {
            return 2 + src_max_value - Value::Card_3 + 1;
        }
        return 0;
    }

    if (src_min_value == Value::Card_2) {
        if (src_max_value == Value::Card_2) {
            return 1;
        } else if (src_max_value >= Value::Card_3) {
            return 1 + src_max_value - Value::Card_3 + 1;
        }
        return 0;
    }

    if (src_min_value >= Value::Card_3) {
        return src_max_value - Value::Card_3 + 1;
    }
    return 0;
}

/*
bool searchSeries_FromA_Or_2(SlotClassify* classify, Card src_min, Card src_max
    , int32_t src_series_count, std::vector<Card>* out)
{
    Value src_min_value = getCardValue(src_min);                    // 目标顺子最小值
    Value src_max_value = getCardValue(src_max);                    // 目标顺子最大值
    int32_t src_series_len = src_max_value - src_min_value + 1;     // 目标顺子长度

    // A2345...
    if (src_min_value == Value::Card_A) {
        // 先选2开始的顺
        SlotClassify classifyBk = *classify;
        std::vector<Card> temp_out{};
        auto ret = searchSeries_From2(&classifyBk, src_series_len, src_series_count, &temp_out);
        if (ret) {
            *classify = std::move(classifyBk);
            *out = std::move(temp_out);
            return true;
        }
    }
    // 选择从3开始
    return searchSeries_From3(classify, Value::Card_3, src_series_len, src_series_count, out);
}
*/

// 从3开始选顺
bool searchSeries_From3(SlotClassify* classify
    , Value src_min_val                     // 从哪张牌开始选。
    , int32_t src_len                   // 长度多少
    , int32_t src_count                 // 每种牌选几张
    , std::vector<Card>* out)
{
    // 顺子最大值是A,没有顺子能压过A
    if (src_min_val + src_len - 1 >= Value::Card_A)
        return false;

    for (Value val = src_min_val; val <= Value::Card_A; val = static_cast<Value>(val + 1)) {
        // 可选顺子的个数不足
        if (Value::Card_A - val + 1 < src_len)
            return false;

        Slot* slot = classify->getSlotByValue(val);
        if (slot->m_total_num >= src_len) {
            // 找到满足条件的第一张牌, 判断后续的牌数量是否足够
            int32_t need_len = 0;
            for (; need_len < src_len; ++need_len) {
                Slot* next_slot = classify->getSlotByValue(static_cast<Value>(val + need_len));
                if (next_slot->m_total_num < src_count) {
                    // 其中一个牌的数量不够
                    break;
                }
            }

            if (need_len == src_len) {
                // 后续的牌数量都足够,选择牌
                for (Value i = val; i < val + need_len; i = static_cast<Value>(i + 1)) {
                    int32_t temp = src_count;
                    Slot* s = classify->getSlotByValue(i);
                    while (temp--) {
                        out->push_back(s->popOneCard());
                    }
                }
                return true;
            }
        }
    }
    return false;
}

// 从2开始选顺长度为src_len的顺
bool searchSeries_From2(SlotClassify* classify
    , int32_t src_len                   // 长度多少
    , int32_t src_series_count          // 每种牌选几张
    , std::vector<Card>* out)
{
    Slot* slot_card_2 = classify->getSlotByValue(Value::Card_2);
    if (slot_card_2->m_total_num < src_series_count)
        return false;

    // 从3开始找
    int32_t need_len = 0;
    for (; need_len < src_len - 1; ++need_len) {
        Slot* slot = classify->getSlotByValue(static_cast<Value>(Value::Card_3 + need_len));
        if (slot->m_total_num < src_series_count) {
            // 其中一个牌的数量不够
            break;
        }
    }

    if (need_len == src_len) {
        // 牌数量足够,选择2
        for (int32_t i = 0; i != src_series_count; ++i) {
            out->push_back(slot_card_2->popOneCard());
        }
        // 选择3开始的顺
        for (Value i = Value::Card_3; i < Value::Card_3 + need_len; i = static_cast<Value>(i + 1)) {
            int32_t temp = src_series_count;
            Slot* s = classify->getSlotByValue(i);
            while (temp--) {
                out->push_back(s->popOneCard());
            }
        }
        return true;
    }
    return false;
}

bool searchBomb_AAA(SlotClassify* classify, Card src_min, std::vector<Card>* out)
{
    Slot* s = classify->getSlotByValue(Value::Card_A);
    if (s->m_total_num == 3) {
        int32_t temp = 3;
        while (temp--) {
            out->push_back(s->popOneCard());
        }
        return true;
    }
    return false;
}

bool searchBomb(SlotClassify* classify, Card src_min, std::vector<Card>* out)
{
    for (Value val = static_cast<Value>(getCardValue(src_min) + 1)
        ; val <= Value::Card_2
        ; val = static_cast<Value>(val + 1)) {
        Slot* s = classify->getSlotByValue(val);
        if (s->m_total_num == 4) {
            int32_t temp = 4;
            while (temp--) {
                out->push_back(s->popOneCard());
            }
            return true;
        }
    }
    return false;
}

} // detail

} // gp_alg
