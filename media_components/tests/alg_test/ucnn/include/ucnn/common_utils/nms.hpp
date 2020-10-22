#ifndef UCNN3_COMMOM_TOOLS_NMS_HPP
#define UCNN3_COMMOM_TOOLS_NMS_HPP

#include <functional>
#include <map>

namespace ucnn
{
template <typename T>
float GetIOU(const T& box1, const T& box2)
{
    float inter;
    {
        float xmin = std::max(box1.xmin, box2.xmin);
        float ymin = std::max(box1.ymin, box2.ymin);
        float xmax = std::min(box1.xmax, box2.xmax);
        float ymax = std::min(box1.ymax, box2.ymax);
        float w = std::max(0.0f, xmax - xmin + 1);
        float h = std::max(0.0f, ymax - ymin + 1);
        inter = w * h;
    }

    float box1_area;
    {
        float w = std::max(0.0f, box1.xmax - box1.xmin + 1);
        float h = std::max(0.0f, box1.ymax - box1.ymin + 1);
        box1_area = w * h;
    }

    float box2_area;
    {
        float w = std::max(0.0f, box2.xmax - box2.xmin + 1);
        float h = std::max(0.0f, box2.ymax - box2.ymin + 1);
        box2_area = w * h;
    }

    float union_area = box1_area + box2_area - inter;
    if (int(union_area) == 0)
    {
        return 0;
    }

    return inter / union_area;
}

template <typename T>
std::vector<T> NMS(const std::vector<T>& boxes, float threshold,
                   const std::function<float(const T&, const T&)>& iou_func = GetIOU<T>)
{
    if (!iou_func)
    {
        printf("iou_func is nullptr.\n");
        return boxes;
    }

    using ScoreMapper = std::multimap<float, int>;
    ScoreMapper sm;
    for (size_t i = 0; i < boxes.size(); i++)
    {
        sm.insert(ScoreMapper::value_type(boxes[i].score, i));
    }

    std::vector<int> pick;
    while (!sm.empty())
    {
        int last_idx = sm.rbegin()->second;
        pick.push_back(last_idx);
        const T& last = boxes[last_idx];
        sm.erase(--sm.end());
        for (auto it = sm.begin(); it != sm.end();)
        {
            int idx = it->second;
            const T& curr = boxes[idx];
            float overlap = iou_func(curr, last);
            if (overlap > threshold)
            {
                auto it_ = it;
                it_++;
                sm.erase(it);
                it = it_;
            }
            else
            {
                it++;
            }
        }
    }

    std::vector<T> ret;
    for (size_t i = 0; i < pick.size(); ++i)
    {
        int index = pick[i];
        ret.push_back(boxes[index]);
    }

    return ret;
}

}  // namespace ucnn

#endif  // UCNN3_COMMOM_TOOLS_NMS_HPP
