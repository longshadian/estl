#ifndef UCNN3_COMMON_TOOLS_PROPOSAL_H
#define UCNN3_COMMON_TOOLS_PROPOSAL_H

#include "ucnn/core/common_types.h"
#include "ucnn/dnn/tensor.hpp"

namespace ucnn
{
struct ProposalParam
{
    int* anchor_scale = nullptr;
    float* variance = nullptr;
    int num_anchors = 0;
    int feat_stride = 0;

    float score_th = 0;

    int net_width = 0;
    int net_height = 0;
    float scale_factor = 0;
};
std::vector<Box> ProposalSSH(const Tensor& scores, const Tensor& boxes, const ProposalParam& param);

}  // namespace ucnn

#endif  // UCNN3_COMMON_TOOLS_PROPOSAL_H
