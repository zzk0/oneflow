#include "oneflow/core/register/lazy_blob.h"
#include "oneflow/core/common/eigen_util.h"

namespace oneflow {

namespace {

const int32_t kSize = 10000000;

template<typename T>
Blob* NewTestBlob(const BlobDesc* blob_desc, T* data) {
  return NewBlob(nullptr, blob_desc, reinterpret_cast<char*>(data), DeviceType::kCPU);
}

Blob* TestBlob() {
  auto* data = new std::vector<int32_t>(kSize, 1);
  auto* blob_desc =
      new BlobDesc(Shape({static_cast<int64_t>(data->size())}), DataType::kInt32, false, false, 1);
  return NewTestBlob(blob_desc, data->data());
}

void LazyEvaluate() {
  Blob* x0_blob = TestBlob();
  Blob* x1_blob = TestBlob();
  Blob* x2_blob = TestBlob();
  Blob* x3_blob = TestBlob();
  Blob* x4_blob = TestBlob();
  Blob* ret_blob = TestBlob();
  int64_t start = GetCurTime();
  LAZY_EVALUATE(int32_t, lazy) {
    lazy(ret_blob) = lazy(x0_blob) * lazy(x1_blob) * (lazy(x0_blob) + lazy(x1_blob))
                     * (lazy(x2_blob) + lazy(x3_blob) + lazy(x4_blob));
  }
  int64_t end = GetCurTime();
  std::cout << "lazy evaluation: " << end - start << std::endl;
}

void EigenEvaluate() {
  Blob* x0_blob = TestBlob();
  Blob* x1_blob = TestBlob();
  Blob* x2_blob = TestBlob();
  Blob* x3_blob = TestBlob();
  Blob* x4_blob = TestBlob();
  Blob* ret_blob = TestBlob();
  ConstEigenArrayMap<int32_t> x0_eigen(x0_blob->dptr<int32_t>(), kSize, 1);
  ConstEigenArrayMap<int32_t> x1_eigen(x1_blob->dptr<int32_t>(), kSize, 1);
  ConstEigenArrayMap<int32_t> x2_eigen(x2_blob->dptr<int32_t>(), kSize, 1);
  ConstEigenArrayMap<int32_t> x3_eigen(x3_blob->dptr<int32_t>(), kSize, 1);
  ConstEigenArrayMap<int32_t> x4_eigen(x4_blob->dptr<int32_t>(), kSize, 1);
  EigenArrayMap<int32_t> ret_eigen(ret_blob->mut_dptr<int32_t>(), kSize, 1);
  int64_t start = GetCurTime();
  ret_eigen = x0_eigen + x1_eigen + (x0_eigen + x1_eigen) + (x2_eigen + x3_eigen + x4_eigen);
  int64_t end = GetCurTime();
  std::cout << "eigen evaluation: " << end - start << std::endl;
}

void EagerEvaluate() {
  int32_t* x0_ptr = TestBlob()->mut_dptr<int32_t>();
  int32_t* x1_ptr = TestBlob()->mut_dptr<int32_t>();
  int32_t* x2_ptr = TestBlob()->mut_dptr<int32_t>();
  int32_t* x3_ptr = TestBlob()->mut_dptr<int32_t>();
  int32_t* x4_ptr = TestBlob()->mut_dptr<int32_t>();
  int32_t* x5_ptr = TestBlob()->mut_dptr<int32_t>();
  int32_t* x6_ptr = TestBlob()->mut_dptr<int32_t>();
  int32_t* x7_ptr = TestBlob()->mut_dptr<int32_t>();
  int32_t* x8_ptr = TestBlob()->mut_dptr<int32_t>();
  int32_t* ret_ptr = TestBlob()->mut_dptr<int32_t>();
  int64_t start = GetCurTime();
  FOR_RANGE(int32_t, i, 0, kSize) { x5_ptr[i] = x0_ptr[i] + x1_ptr[i]; }
  FOR_RANGE(int32_t, i, 0, kSize) { x6_ptr[i] = x2_ptr[i] + x3_ptr[i]; }
  FOR_RANGE(int32_t, i, 0, kSize) { x7_ptr[i] = x6_ptr[i] + x4_ptr[i]; }
  FOR_RANGE(int32_t, i, 0, kSize) { x8_ptr[i] = x1_ptr[i] * x7_ptr[i]; }
  FOR_RANGE(int32_t, i, 0, kSize) { ret_ptr[i] = x2_ptr[i] * x8_ptr[i]; }
  int64_t end = GetCurTime();
  std::cout << "eager evaluation: " << end - start << std::endl;
}

ALWAYS_INLINE int32_t Add(const int32_t x, const int32_t y) { return x + y; }

struct Adder final {
  static int32_t Add(const int32_t x, const int32_t y) { return x + y; }
};

void TestAdder() {
  int32_t* x0_ptr = TestBlob()->mut_dptr<int32_t>();
  int32_t* x1_ptr = TestBlob()->mut_dptr<int32_t>();

  {
    int64_t start = GetCurTime();
    FOR_RANGE(int32_t, i, 0, kSize) { x1_ptr[i] = Add(x0_ptr[i], x0_ptr[i]); }
    int64_t end = GetCurTime();
    std::cout << "Add: " << end - start << std::endl;
  }
  {
    int64_t start = GetCurTime();
    FOR_RANGE(int32_t, i, 0, kSize) { x1_ptr[i] = Adder::Add(x0_ptr[i], x0_ptr[i]); }
    int64_t end = GetCurTime();
    std::cout << "Adder::Add: " << end - start << std::endl;
  }
}

void LazyBlobPerformance() {
  EagerEvaluate();
  EigenEvaluate();
  LazyEvaluate();
  TestAdder();
}

}  // namespace

}  // namespace oneflow

int main() {
  oneflow::LazyBlobPerformance();
  return 0;
}
