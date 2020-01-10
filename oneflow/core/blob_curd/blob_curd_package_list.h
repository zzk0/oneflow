#ifndef ONEFLOW_CORE_BLOB_CURD_BLOB_CURD_PACKAGE_LIST_H_
#define ONEFLOW_CORE_BLOB_CURD_BLOB_CURD_PACKAGE_LIST_H_

#include "oneflow/core/blob_curd/blob_curd.h"

namespace oneflow {

template<typename RequestOrResponse>
class BlobCurdPackageList final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlobCurdPackageList);
  explicit BlobCurdPackageList(size_t package_limit) : package_limit_(package_limit) {}
  ~BlobCurdPackageList() = default;

  size_t package_limit() const { return package_limit_; }

  RequestOrResponse* AddItem(size_t data_byte_size) { TODO(); }
  void ForEachPackage(std::function<void(std::shared_ptr<std::vector<char>>)> Handler) const { TODO(); }

 private:
  const size_t package_limit_;
  std::list<std::shared_ptr<std::vector<char>>> packages_;
};

}

#endif  // ONEFLOW_CORE_BLOB_CURD_BLOB_CURD_PACKAGE_LIST_H_
