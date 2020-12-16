/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef ONEFLOW_API_PYTHON_FRAMEWORK_EXCEPTION_H_
#define ONEFLOW_API_PYTHON_FRAMEWORK_EXCEPTION_H_

#include <string>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/exception.h"

using namespace oneflow;

Maybe<void> ToDoError() { TODO_THEN_RETURN(); }

Maybe<void> UnImplError() { UNIMPLEMENTED_THEN_RETURN(); }

void TestErrorException(std::string error_type) {
  using namespace oneflow;
  cfg::ErrorProto error_;

  error_.set_error_summary("test error exception");
  error_.set_msg("TestErrorException");
  error_.mutable_check_failed_error();

  if (error_type == "todo") {
    auto maybe_obj = ToDoError();
    throw TodoException(*maybe_obj.GetDataAndErrorProto());
  } else if (error_type == "unimpl") {
    auto maybe_obj = UnImplError();
    throw UnimplementedException(*maybe_obj.GetDataAndErrorProto());
  } else if (error_type == "error") {
    if (error_.has_error_type()) {
      ErrorException exp(error_);
      throw exp;
    }
  }
}

#endif  // ONEFLOW_API_PYTHON_FRAMEWORK_EXCEPTION_H_
