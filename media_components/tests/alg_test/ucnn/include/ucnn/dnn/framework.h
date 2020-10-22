#ifndef UCNN_DNN_FRAMEWORK_H
#define UCNN_DNN_FRAMEWORK_H

#include "ucnn/core/mat.h"
#include "ucnn/dnn/dnn_types.h"
#include "ucnn/dnn/tensor.hpp"

namespace ucnn
{
class BaseFramework
{
  public:
    virtual ~BaseFramework(){};

    virtual void Forward(const Mat &mat){};

    virtual void Forward(const std::vector<Mat> &mats){};

    virtual std::shared_ptr<Tensor> GetOutputByName(const std::string &out_layer, bool copy = true) { return {}; };
};
}  // namespace ucnn

#endif  // UCNN_DNN_FRAMEWORK_H
