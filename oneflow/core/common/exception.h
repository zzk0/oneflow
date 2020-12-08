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
#ifndef ONEFLOW_CORE_COMMON_ERROR_EXPECTION_H_
#define ONEFLOW_CORE_COMMON_ERROR_EXPECTION_H_

#include <exception>
#include <cassert>
#include <string>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <limits.h>
#include "oneflow/core/common/error.cfg.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

std::string debug_info(const char* file_name);

class OneflowException : public std::exception {
 public:
  virtual const char* what() const throw() { return "Oneflow exception happened"; }
};

class BaseException : public OneflowException {
 public:
  BaseException(const std::string& msg) : msg_{msg} {}

  virtual const char* what() const throw() { return msg_.data(); }

 private:
  std::string msg_;
};

class ErrorException : public OneflowException {
 public:
  explicit ErrorException(const cfg::ErrorProto& error_proto);

  const char* what() const noexcept override {
    std::string res = "\n\nerror msg: \n\n" + error_summary_ + "\n";
    if (message_.has_op_kernel_not_found_error()) {
      res = res + get_op_kernel_not_found_error_str();
      // message_.clear_op_kernel_not_found_error();
    } else if (message_.has_multiple_op_kernels_matched_error()) {
      res = res + get_multiple_op_kernels_matched_error_str();
      // message_.clear_multiple_op_kernels_matched_error();
    }
    res = res + message_.DebugString();
    res = res + msg_;
    return res.data();
  }

 private:
  std::string get_op_kernel_not_found_error_str() const;
  std::string get_multiple_op_kernels_matched_error_str() const;

  cfg::ErrorProto message_;
  std::string error_summary_;
  std::string msg_;
};

class UnimplementedException : public OneflowException {
 public:
  explicit UnimplementedException(const std::string& debug_info)
      : info_("\nOneflow exception: UNIMPLEMENTED") {
    info_ = info_ = info_ + debug_info;
  }

  const char* what() const noexcept override { return info_.data(); }

 private:
  std::string info_;
};

class TodoException : public OneflowException {
 public:
  explicit TodoException(const std::string& debug_info) : info_("\nOneflow exception: TODO") {
    info_ = info_ + debug_info;
  }

  const char* what() const noexcept override { return info_.data(); }

 private:
  std::string info_;
};

}  // namespace oneflow

#define DEBUG_INFO std::string("\nFile: ") + debug_info(__FILE__) + ": " + std::to_string(__LINE__)

#endif  // ONEFLOW_CORE_COMMON_ERROR_EXPECTION_H_
