#ifndef UCNN3_COMMOM_TOOLS_ASSIGN_H
#define UCNN3_COMMOM_TOOLS_ASSIGN_H

#include <memory>
#include <tuple>
#include <vector>

namespace ucnn
{
using AssignResult = std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>;

class Munkres;
class AssignSolver
{
  public:
    enum AssignType
    {
        Hungarian,
        Greedy,
    };
    enum MatchType
    {
        LARGER_MATCH,
        LESS_MATCH
    };
    AssignSolver(AssignType assign_type = Hungarian);
    ~AssignSolver();

    struct AssignParam
    {
        double match_threshold;
        MatchType match_type;
    };

    template <typename T>
    AssignResult AssignDetectionsToTracks(const std::vector<T>& match_scores, int track_number, int detect_number,
                                          const AssignParam& param);

  private:
    AssignType assign_type_;

    std::unique_ptr<Munkres> munkres_;

    void Init();

    template <typename T>
    std::vector<std::pair<int, int>> Compute(const std::vector<T>& match_scores, int rows, int cols,
                                             const AssignParam& param);
};

/// Implementation
template <typename T>
AssignResult AssignSolver::AssignDetectionsToTracks(const std::vector<T>& match_scores, int track_number,
                                                    int detect_number, const AssignParam& param)
{
    /*
            COST |   D0   |   D1   |   D2   |   D3   |
            --------------------------------------------
            T0  | roi_00 | roi_01 | roi_02 | roi_03 |
            --------------------------------------------
            T1  | roi_10 | roi_11 | roi_12 | roi_13 |
            --------------------------------------------
            T2  | roi_20 | roi_21 | roi_22 | roi_23 |
            --------------------------------------------
            */

    std::vector<std::pair<int, int>> assignments;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;

    if (match_scores.empty())
    {
        return {};
    }

    std::vector<int> track_bin(track_number, 1);
    std::vector<int> detection_bin(detect_number, 1);

    assignments = Compute(match_scores, track_number, detect_number, param);
    for (size_t i = 0; i < assignments.size(); i++)
    {
        track_bin[assignments[i].first] = 0;
        detection_bin[assignments[i].second] = 0;
    }
    for (int i = 0; i < track_number; i++)
        if (track_bin[i])
            unmatched_tracks.push_back(i);
    for (int i = 0; i < detect_number; i++)
        if (detection_bin[i])
            unmatched_detections.push_back(i);
    return std::make_tuple(assignments, unmatched_tracks, unmatched_detections);
}

}  // namespace ucnn

#endif  // UCNN3_COMMOM_TOOLS_ASSIGN_H
