#ifndef ONEFLOW_CORE_AUTO_PLACEMNENT_DF_FUNC_H_
#define ONEFLOW_CORE_AUTO_PLACEMNENT_DF_FUNC_H_

#include "oneflow/core/auto_placement/df_value.h"

namespace oneflow {

DfValue WeightVar(std::shared_ptr<Value> value, double lr);

inline DfValue WeightVar(std::shared_ptr<Value> value) {
  return WeightVar(value, 0.01);
}

DfValue Square(DfValue input);

DfValue Backward(DfValue loss);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PLACEMNENT_DF_FUNC_H_
