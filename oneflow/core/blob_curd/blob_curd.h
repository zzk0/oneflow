#ifndef ONEFLOW_CORE_BLOB_CURD_BLOB_CURD_H_
#define ONEFLOW_CORE_BLOB_CURD_BLOB_CURD_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/pod_proto.h"

namespace oneflow {

#define BLOB_CURD_METHOD_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(invalide_curd) \
  OF_PP_MAKE_TUPLE_SEQ(fill_last_tensor)

#define BLOB_CURD_TYPE(method) k_##method##_BlobCurdType

enum BlobCurdType {
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
  POD_PROTO_DEFINE_FIELD(int64_t, mem_zone_id);
  POD_PROTO_DEFINE_FIELD(int64_t, global_blob_id);
  POD_PROTO_DEFINE_FIELD(int64_t, offset);
};

template<template <BlobCurdType> class BlobCurdParam>
struct BlobCurdHeader {
  POD_PROTO_DEFINE_FIELD(int64_t, token);
  POD_PROTO_DEFINE_FIELD(int64_t, machine_id);
  POD_PROTO_DEFINE_FIELD(int64_t, data_byte_size);

  size_t total_byte_size() const { return sizeof(*this) + data_byte_size(); }

#define MAKE_ONEOF_CASE(method) POD_PROTO_ONEOF_CASE(BlobCurdParam<BLOB_CURD_TYPE(method)>, method)
  POD_PROTO_DEFINE_FIELD(BlobCurdType, POD_PROTO_ONEOF_CASE(curd_type));
  POD_PROTO_DEFINE_ONEOF_UNION(OF_PP_FOR_EACH_TUPLE(MAKE_ONEOF_CASE, BLOB_CURD_METHOD_SEQ));
  POD_PROTO_DEFINE_ONEOF_ACCESSOR(BLOB_CURD_TYPE, curd_type,
                                  OF_PP_FOR_EACH_TUPLE(MAKE_ONEOF_CASE, BLOB_CURD_METHOD_SEQ));
#undef MAKE_ONEOF_CASE
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
  POD_PROTO_DEFINE_FIELD(int64_t, package_byte_limit);
  POD_PROTO_DEFINE_FIELD(int64_t, items_byte_size);
  POD_PROTO_DEFINE_FIELD(int64_t, num_items);
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
