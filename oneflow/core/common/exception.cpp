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
#include "oneflow/core/common/exception.h"

namespace oneflow {

namespace {

inline std::string strip(const std::string& str, char ch = '\0') {
  int64_t start = 0;
  int64_t end = str.size() - 1;
  int64_t len = str.size();
  if (len == 0) { return std::string(); }
  if (ch == '\0') {
    while (str[start] == ' ' || str[start] == '\n') {
      ++start;
      if (start == len) { break; }
    }
    while (str[end] == ' ' || str[end] == '\n') {
      --end;
      if (end == -1) { break; }
    }
  } else {
    while (str[start] == ch) { ++start; }
    while (str[end] == ch) { --end; }
  }
  if (end < start) {
    return std::string();
  } else {
    return str.substr(start, end - start + 1);
  }
}

inline std::vector<std::string> split(const std::string& str, char ch = '\n') {
  std::vector<std::string> res;
  int64_t start = 0;
  int64_t pos = 0;
  int64_t len = str.size();
  if (len == 0) { return res; }
  while (pos < len) {
    if (str[start] == ch) {
      res.emplace_back(str.substr(start, pos - start));
      start = ++pos;
      continue;
    }
    ++pos;
  }
  if (start < pos) { res.emplace_back(str.substr(start, pos - start + 1)); }
  return res;
}

inline std::string join(const std::vector<std::string>& strs, char ch = '\n') {
  if (strs.empty()) { return std::string(); }
  std::string res;
  for (std::string elem : strs) { res = res + elem + ch; }
  if (!res.empty()) { res.pop_back(); }
  return res;
}

}  // namespace

std::string debug_info(const char* file_name) {
  char abs_path_buffer[PATH_MAX];
  if (realpath(file_name, abs_path_buffer)) {
    return std::string(abs_path_buffer);
  } else {
    std::string err_msg = std::string("Get absolute path failed: ") + file_name;
    throw BaseException(err_msg);
  }
}

ErrorException::ErrorException(const cfg::ErrorProto& error_proto) : message_{error_proto} {
  assert(message_.has_error_type());
  error_summary_ = message_.error_summary();
  message_.clear_error_summary();
  msg_ = message_.msg();
  message_.clear_msg();
  // if (std::getenv("ONEFLOW_DEBUG_MODE") == nullptr
  //     && (!Global<ResourceDesc, ForSession>::Get()
  //         || !Global<ResourceDesc, ForSession>::Get()->enable_debug_mode())) {
  //   message_.clear_stack_frame();
  // }
}

std::string ErrorException::get_op_kernel_not_found_error_str() const {
  std::string error_msg = message_.op_kernel_not_found_error().DebugString();
  // simulate error_msg = error_msg.replace("\\", "")
  error_msg.erase(std::remove(error_msg.begin(), error_msg.end(), '\\'), error_msg.end());
  std::string substr_need_del = "op_kernels_not_found_debug_str:";
  // simulate error_msg = error_msg.replace("op_kernels_not_found_debug_str:", "")
  while (error_msg.find(substr_need_del) < error_msg.length()) {
    error_msg.erase(error_msg.find(substr_need_del), substr_need_del.size());
  }
  // simulate error_msg = "\n".join([e.strip()[1:-1] for e in error_msg.strip().split("\n")])
  error_msg = strip(error_msg);
  std::vector<std::string> splited_msg = split(error_msg);
  std::vector<std::string> processed_splited_msg;
  for (std::string elem : splited_msg) {
    std::string msg = strip(elem);
    if (!msg.empty()) { msg = msg.substr(1); }
    if (!msg.empty()) { msg.pop_back(); }
    processed_splited_msg.emplace_back(msg);
  }
  error_msg = join(processed_splited_msg, '\n');
  error_msg = "\n\nFailure messages of registered kernels for current Op node: \n" + error_msg;
  return error_msg;
}

std::string ErrorException::get_multiple_op_kernels_matched_error_str() const {
  std::string error_msg = message_.multiple_op_kernels_matched_error().DebugString();
  error_msg.erase(std::remove(error_msg.begin(), error_msg.end(), '\\'), error_msg.end());
  std::string substr_need_del = "matched_op_kernels_debug_str:";
  while (error_msg.find(substr_need_del) < error_msg.length()) {
    error_msg.erase(error_msg.find(substr_need_del), substr_need_del.size());
  }
  error_msg = strip(error_msg);
  std::vector<std::string> splited_msg = split(error_msg);
  std::vector<std::string> processed_splited_msg;
  for (std::string elem : splited_msg) {
    std::string msg = strip(elem);
    if (!msg.empty()) { msg = msg.substr(1); }
    if (!msg.empty()) { msg.pop_back(); }
    processed_splited_msg.emplace_back(msg);
  }
  error_msg = join(processed_splited_msg, '\n');
  error_msg =
      "\n\nThere exists multiple registered kernel candidates for current Op node: \n" + error_msg;
  return error_msg;
}

}  // namespace oneflow
