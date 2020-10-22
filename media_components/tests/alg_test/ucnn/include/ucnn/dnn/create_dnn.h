#ifndef UCNN_CREATE_DNN_H
#define UCNN_CREATE_DNN_H

#include "ucnn/dnn/framework.h"

namespace ucnn
{
std::shared_ptr<BaseFramework> CreateUcnnPtr(const ModelParams& model_params);

}  // namespace ucnn

#endif  // UCNN_CREATE_DNN_H
