#ifndef ONEFLOW_CORE_BLOB_CURD_BLOB_CURD_H_
#define ONEFLOW_CORE_BLOB_CURD_BLOB_CURD_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

#define BLOB_CURD_METHOD_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(fill_last_tensor)

#define BLOB_CURD_TYPE(method) k_##method##_BlobCurdType

#define DEFINE_FIELD_AND_ACCESSOR(type, field_name)        \
 public:                                                   \
  type field_name() const { return field_name##_; }        \
  void set_##field_name(type val) { field_name##_ = val; } \
                                                           \
 private:                                                  \
  type field_name##_;

enum BlobCurdType {
  kInvalidBlobCurdType = 0,
#define MAKE_BLOB_CURD_TYPE(method) BLOB_CURD_TYPE(method),
  OF_PP_FOR_EACH_TUPLE(MAKE_BLOB_CURD_TYPE, BLOB_CURD_METHOD_SEQ)
#undef MAKE_BLOB_CURD_TYPE
};

template<BlobCurdType curd_type>
struct BlobCurdRequestParam {};

template<BlobCurdType curd_type>
struct BlobCurdResponseParam {};

template<>
struct BlobCurdRequestParam<BLOB_CURD_TYPE(fill_last_tensor)> final {
  DEFINE_FIELD_AND_ACCESSOR(int64_t, offset);
};

template<template <BlobCurdType> class BlobCurdParam>
struct BlobCurdHeader {
  DEFINE_FIELD_AND_ACCESSOR(int64_t, token);
  DEFINE_FIELD_AND_ACCESSOR(int64_t, machine_id);
  DEFINE_FIELD_AND_ACCESSOR(int64_t, mem_zone_id);
  DEFINE_FIELD_AND_ACCESSOR(int64_t, global_blob_id);
  DEFINE_FIELD_AND_ACCESSOR(int64_t, data_byte_size);

  size_t total_byte_size() const { return sizeof(*this) + data_byte_size(); }

  // a simulation to protobuf's oneof utils
  // eg. 1) blob_curd_header.fill_last_tensor()
  //     2) blob_curd_header->mutable_fill_last_tensor()
  DEFINE_FIELD_AND_ACCESSOR(BlobCurdType, curd_type_case);
 public:  
#define DEFINE_CURD_ONEOF_CASE(method)                                  \
  const BlobCurdParam<BLOB_CURD_TYPE(method)>& method() const {  \
    CHECK_EQ(curd_type_case(), BLOB_CURD_TYPE(method));        \
    return method_;                                                     \
  } \
  BlobCurdParam<BLOB_CURD_TYPE(method)>* mutable_##method() {  \
    set_curd_type_case(BLOB_CURD_TYPE(method)); \
    return &method_;                                              \
  }
  OF_PP_FOR_EACH_TUPLE(DEFINE_CURD_ONEOF_CASE, BLOB_CURD_METHOD_SEQ);
#undef DEFINE_CURD_ONEOF_CASE

 private:
  union {
#define DECLEAR_CURD_ONEOF_CASE(x) BlobCurdParam<BLOB_CURD_TYPE(x)> x##_;
  OF_PP_FOR_EACH_TUPLE(DECLEAR_CURD_ONEOF_CASE, BLOB_CURD_METHOD_SEQ)
#undef DECLEAR_CURD_ONEOF_CASE
  };
};

using BlobCurdRequestHeader = BlobCurdHeader<BlobCurdRequestParam>;
using BlobCurdResponseHeader = BlobCurdHeader<BlobCurdResponseParam>;

struct BlobCurdRequest {
  BlobCurdRequestHeader header;
  char data[0];
};

struct BlobCurdResponse {
  BlobCurdResponseHeader header;
  char data[0];
};

struct BlobCurdPackageHeader {
  size_t package_byte_limit;
  size_t items_byte_size;
  size_t num_items;
};

template<typename RequestOrResponse>
struct BlobCurdPackage {
  BlobCurdPackageHeader header;
  char data[0];

  size_t package_byte_size() const { return sizeof(BlobCurdPackage) + header.items_byte_size; }
  RequestOrResponse* AddItem(size_t data_byte_size) { TODO(); }
  void ForEachItem(std::function<void(const RequestOrResponse&)> Handler) const { TODO(); }
};

}

#endif  // ONEFLOW_CORE_BLOB_CURD_BLOB_CURD_H_
