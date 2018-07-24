#ifndef ONEFLOW_CORE_REGISTER_LAZY_BLOB_H_
#define ONEFLOW_CORE_REGISTER_LAZY_BLOB_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

#if defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline inline
#elif defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define ALWAYS_INLINE inline
#endif

class LazyBlobNode {
 public:
  virtual ~LazyBlobNode() = default;

  void Evaluate() const {
    if (backend_blob()) { Evaluate(nullptr); }
  }
  virtual void Evaluate(Blob* output_blob) const = 0;

  // Getters
  const Blob* backend_blob() const { return backend_blob_; }
  const Shape& shape() const { return shape_; }
  DataType data_type() const { return data_type_; }

 protected:
  LazyBlobNode(const Shape& shape, DataType data_type)
      : backend_blob_(nullptr), shape_(shape), data_type_(data_type) {}
  LazyBlobNode(Blob* backend_blob)
      : backend_blob_(backend_blob),
        shape_(backend_blob->shape()),
        data_type_(backend_blob->data_type()) {}

  Blob* mut_backend_blob() const { return backend_blob_; }

 private:
  Blob* backend_blob_;
  Shape shape_;
  DataType data_type_;
};

template<typename DerivedT>
class LazyBlobIf : public LazyBlobNode {
 public:
  const static bool is_element_wise = false;
  virtual ~LazyBlobIf() = default;

  virtual void Evaluate(Blob* output_blob) const override {
    CHECK(this->backend_blob() == nullptr);
    switch (this->shape().NumAxes()) {
      case 1: return Evaluate1(output_blob);
      case 2: return Evaluate2(output_blob);
      case 3: return Evaluate3(output_blob);
      case 4: return Evaluate4(output_blob);
      case 5: return Evaluate5(output_blob);
      default: UNIMPLEMENTED();
    }
  }

 protected:
  LazyBlobIf(const Shape& shape, DataType data_type) : LazyBlobNode(shape, data_type) {}
  LazyBlobIf(Blob* backend_blob) : LazyBlobNode(backend_blob) {}

 private:
  void Evaluate1(Blob* output_blob) const;
  void Evaluate2(Blob* output_blob) const;
  void Evaluate3(Blob* output_blob) const;
  void Evaluate4(Blob* output_blob) const;
  void Evaluate5(Blob* output_blob) const;
};

template<typename T>
class VarLazyBlob final : public LazyBlobIf<VarLazyBlob<T>> {
 public:
  typedef T dtype;
  const static bool is_element_wise = true;

  VarLazyBlob(Blob* backend_blob)
      : LazyBlobIf<VarLazyBlob<T>>(backend_blob),
        dptr_(backend_blob->mut_dptr<T>()),
        dim0_next_dim_count_(ShapeDefaultedNextDimCount(backend_blob->shape(), 0)),
        dim1_next_dim_count_(ShapeDefaultedNextDimCount(backend_blob->shape(), 1)),
        dim2_next_dim_count_(ShapeDefaultedNextDimCount(backend_blob->shape(), 2)),
        dim3_next_dim_count_(ShapeDefaultedNextDimCount(backend_blob->shape(), 3)),
        dim4_next_dim_count_(ShapeDefaultedNextDimCount(backend_blob->shape(), 4)) {
    CHECK_EQ(GetDataType<T>::value, backend_blob->data_type());
  }

  VarLazyBlob<T>& operator=(const LazyBlobNode& value_lazy_blob_node) {
    CHECK(value_lazy_blob_node.backend_blob() == nullptr);
    CHECK(this->shape() == value_lazy_blob_node.shape());
    value_lazy_blob_node.Evaluate(this->mut_backend_blob());
    return *this;
  }

  void Evaluate(Blob* output_blob) const override {
    // Do nothing
  }

  ALWAYS_INLINE dtype At(int64_t dim0) const { return dptr_[dim0]; }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1) const {
    return dptr_[dim0 * dim0_next_dim_count_ + dim1];
  }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return dptr_[dim0 * dim0_next_dim_count_ + dim1 * dim1_next_dim_count_ + dim2];
  }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return dptr_[dim0 * dim0_next_dim_count_ + dim1 * dim1_next_dim_count_
                 + dim2 * dim2_next_dim_count_ + dim3];
  }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                         int64_t dim4) const {
    return dptr_[dim0 * dim0_next_dim_count_ + dim1 * dim1_next_dim_count_
                 + dim2 * dim2_next_dim_count_ + dim3 * dim3_next_dim_count_ + dim4];
  }

 private:
  int64_t ShapeDefaultedNextDimCount(const Shape& shape, int32_t index) const {
    CHECK_GE(index, 0);
    return (index + 1 < shape.NumAxes() ? shape.Count(index + 1) : MaxVal<int32_t>());
  };
  T* dptr_;
  const int64_t dim0_next_dim_count_;
  const int64_t dim1_next_dim_count_;
  const int64_t dim2_next_dim_count_;
  const int64_t dim3_next_dim_count_;
  const int64_t dim4_next_dim_count_;
};

template<template<typename> class CoreFunc, typename XT>
class UnaryExpresionLazyBlob final : public LazyBlobIf<UnaryExpresionLazyBlob<CoreFunc, XT>> {
 public:
  using T = typename XT::dtype;
  typedef decltype(CoreFunc<T>::Invoke(*(const T*)nullptr)) dtype;
  const static bool is_element_wise = true;

  explicit UnaryExpresionLazyBlob(XT&& x)
      : LazyBlobIf<UnaryExpresionLazyBlob<CoreFunc, XT>>(x.shape(), GetDataType<dtype>::value),
        x_(x) {}

  ALWAYS_INLINE dtype At(int64_t dim0) const { return CoreFunc<T>::Invoke(x_.At(dim0)); }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1) const {
    return CoreFunc<T>::Invoke(x_.At(dim0, dim1));
  }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return CoreFunc<T>::Invoke(x_.At(dim0, dim1, dim2));
  }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return CoreFunc<T>::Invoke(x_.At(dim0, dim1, dim2, dim3));
  }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                         int64_t dim4) const {
    return CoreFunc<T>::Invoke(x_.At(dim0, dim1, dim2, dim3, dim4));
  }

  virtual typename std::enable_if<XT::is_element_wise>::type Evaluate(
      Blob* output_blob) const override {
    int64_t elem_cnt = this->shape().elem_cnt();
    T* dptr = output_blob->mut_dptr<T>();
    FOR_RANGE(int64_t, i, 0, elem_cnt) { dptr[i] = CoreFunc<T>::Invoke(x_.At(i)); }
  }

 private:
  const XT& x_;
};

template<template<typename> class CoreFunc, typename XT, typename YT,
         typename = typename std::enable_if<
             std::is_same<typename XT::dtype, typename YT::dtype>::value>::type>
class BinaryExpresionLazyBlob final : public LazyBlobIf<BinaryExpresionLazyBlob<CoreFunc, XT, YT>> {
 public:
  using T = typename XT::dtype;
  typedef decltype(CoreFunc<T>::Invoke(*(const T*)nullptr, *(const T*)nullptr)) dtype;
  const static bool is_element_wise = true;

  BinaryExpresionLazyBlob(XT&& x, YT&& y)
      : LazyBlobIf<BinaryExpresionLazyBlob<CoreFunc, XT, YT>>(x.shape(), GetDataType<dtype>::value),
        x_(x),
        y_(y) {
    CHECK(x.shape() == y.shape());
  }

  ALWAYS_INLINE dtype At(int64_t dim0) const {
    return CoreFunc<T>::Invoke(x_.At(dim0), y_.At(dim0));
  }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1) const {
    return CoreFunc<T>::Invoke(x_.At(dim0, dim1), y_.At(dim0, dim1));
  }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return CoreFunc<T>::Invoke(x_.At(dim0, dim1, dim2), y_.At(dim0, dim1, dim2));
  }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return CoreFunc<T>::Invoke(x_.At(dim0, dim1, dim2, dim3), y_.At(dim0, dim1, dim2, dim3));
  }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                         int64_t dim4) const {
    return CoreFunc<T>::Invoke(x_.At(dim0, dim1, dim2, dim3, dim4),
                               y_.At(dim0, dim1, dim2, dim3, dim4));
  }

  virtual typename std::enable_if<XT::is_element_wise && YT::is_element_wise>::type Evaluate(
      Blob* output_blob) const override {
    int64_t elem_cnt = this->shape().elem_cnt();
    T* dptr = output_blob->mut_dptr<T>();
    FOR_RANGE(int64_t, i, 0, elem_cnt) { dptr[i] = CoreFunc<T>::Invoke(x_.At(i), y_.At(i)); }
  }

  ALWAYS_INLINE const XT& x() { return x_; }
  ALWAYS_INLINE const YT& y() { return y_; }

 private:
  const XT& x_;
  const YT& y_;
};

template<typename XT>
class BroadcastLazyBlob final : public LazyBlobIf<BroadcastLazyBlob<XT>> {
 public:
  typedef typename XT::dtype dtype;

  BroadcastLazyBlob(XT&& x, const Shape& shape)
      : LazyBlobIf<BroadcastLazyBlob<XT>>(shape, GetDataType<dtype>::value),
        x_(x),
        dim0_size_(DefaultedShapeAt(0)),
        dim1_size_(DefaultedShapeAt(1)),
        dim2_size_(DefaultedShapeAt(2)),
        dim3_size_(DefaultedShapeAt(3)),
        dim4_size_(DefaultedShapeAt(4)) {
    CheckShape(x.shape(), shape);
  }

  ALWAYS_INLINE dtype At(int64_t dim0) const { return x_.At(dim0 % dim0_size_); }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1) const {
    return x_.At(dim0 % dim0_size_, dim1 % dim1_size_);
  }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return x_.At(dim0 % dim0_size_, dim1 % dim1_size_, dim2 % dim2_size_);
  }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return x_.At(dim0 % dim0_size_, dim1 % dim1_size_, dim2 % dim2_size_, dim3 % dim3_size_);
  }
  ALWAYS_INLINE dtype At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                         int64_t dim4) const {
    return x_.At(dim0 % dim0_size_, dim1 % dim1_size_, dim2 % dim2_size_, dim3 % dim3_size_,
                 dim4 % dim4_size_);
  }

 private:
  void CheckShape(const Shape& small_shape, const Shape& big_shape) {
    CHECK_EQ(small_shape.NumAxes(), big_shape.NumAxes());
    FOR_RANGE(int, i, 0, small_shape.NumAxes()) {
      CHECK_EQ(big_shape.At(i) % small_shape.At(i), 0);
    }
  }
  int64_t DefaultedShapeAt(const Shape& shape, int32_t index) const {
    CHECK_GE(index, 0);
    return (index < shape.NumAxes() ? shape.At(index) : MaxVal<int32_t>());
  };
  const XT& x_;
  const int64_t dim0_size_;
  const int64_t dim1_size_;
  const int64_t dim2_size_;
  const int64_t dim3_size_;
  const int64_t dim4_size_;
};

template<typename T>
class LazyBlobBuilder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyBlobBuilder);
  LazyBlobBuilder() = default;

  template<typename DT = T>
  VarLazyBlob<DT> operator()(Blob* blob) {
    return VarLazyBlob<DT>(blob);
  }
};

#define LAZY_BLOB_BINARY_CORE_OP_FUNC_SEQ    \
  OF_PP_MAKE_TUPLE_SEQ(Add, +, T)            \
  OF_PP_MAKE_TUPLE_SEQ(Sub, -, T)            \
  OF_PP_MAKE_TUPLE_SEQ(Mul, *, T)            \
  OF_PP_MAKE_TUPLE_SEQ(Div, /, T)            \
  OF_PP_MAKE_TUPLE_SEQ(Mod, %, T)            \
  OF_PP_MAKE_TUPLE_SEQ(Eq, ==, bool)         \
  OF_PP_MAKE_TUPLE_SEQ(Ne, !=, bool)         \
  OF_PP_MAKE_TUPLE_SEQ(Gt, >, bool)          \
  OF_PP_MAKE_TUPLE_SEQ(Ge, >=, bool)         \
  OF_PP_MAKE_TUPLE_SEQ(Lt, <, bool)          \
  OF_PP_MAKE_TUPLE_SEQ(Le, <=, bool)         \
  OF_PP_MAKE_TUPLE_SEQ(LogicalAnd, &&, bool) \
  OF_PP_MAKE_TUPLE_SEQ(LogicalOr, &&, bool)

#define DECLARE_LAZY_BLOB_BINARY_CORE(name, op, ret_type)                         \
  template<typename T>                                                            \
  struct LazyBlobCore##name final {                                               \
    static ALWAYS_INLINE ret_type Invoke(const T x, const T y) { return x op y; } \
  };
OF_PP_FOR_EACH_TUPLE(DECLARE_LAZY_BLOB_BINARY_CORE, LAZY_BLOB_BINARY_CORE_OP_FUNC_SEQ);
#undef DECLARE_LAZY_BLOB_BINARY_CORE

#define LAZY_BLOB_UNARY_CORE_OP_FUNC_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(Negative, -, T)   \
  OF_PP_MAKE_TUPLE_SEQ(LogicalNot, !, bool)

template<typename T>
ALWAYS_INLINE T Add(const T x, const T y) {
  return x + y;
}

#define DECLARE_LAZY_BLOB_UNARY_CORE(name, op, ret_type)             \
  template<typename T>                                               \
  struct LazyBlobCore##name final {                                  \
    static ALWAYS_INLINE ret_type Invoke(const T x) { return op x; } \
  };
OF_PP_FOR_EACH_TUPLE(DECLARE_LAZY_BLOB_UNARY_CORE, LAZY_BLOB_UNARY_CORE_OP_FUNC_SEQ);
#undef DECLARE_LAZY_BLOB_UNARY_CORE

template<template<typename> class LazyBlobCoreFunc, typename XT, typename YT = XT>
typename std::enable_if<std::is_base_of<LazyBlobNode, XT>::value
                            && std::is_base_of<LazyBlobNode, YT>::value,
                        BinaryExpresionLazyBlob<LazyBlobCoreFunc, XT, YT>>::type
BuildBinaryLazyBlob(XT&& x, YT&& y) {
  return BinaryExpresionLazyBlob<LazyBlobCoreFunc, XT, YT>(std::move(x), std::move(y));
}

#define OVERLOAD_BINARY_LAZY_BLOB_OP_FUNC(name, op, ret_type)                           \
  template<typename XType, typename YType = XType,                                      \
           typename XT = typename std::remove_reference<XType>::type,                   \
           typename YT = typename std::remove_reference<YType>::type>                   \
  typename std::enable_if<std::is_base_of<LazyBlobNode, XT>::value                      \
                              && std::is_base_of<LazyBlobNode, YT>::value,              \
                          BinaryExpresionLazyBlob<LazyBlobCore##name, XT, YT>>::type    \
  operator op(XType&& x, YType&& y) {                                                   \
    return BuildBinaryLazyBlob<LazyBlobCore##name, XT, YT>(std::move(x), std::move(y)); \
  }
OF_PP_FOR_EACH_TUPLE(OVERLOAD_BINARY_LAZY_BLOB_OP_FUNC, LAZY_BLOB_BINARY_CORE_OP_FUNC_SEQ);
#undef OVERLOAD_BINARY_LAZY_BLOB_OP_FUNC

template<template<typename> class LazyBlobCoreFunc, typename XT>
typename std::enable_if<std::is_base_of<LazyBlobNode, XT>::value,
                        UnaryExpresionLazyBlob<LazyBlobCoreFunc, XT>>::type
BuildUnaryLazyBlob(XT&& x) {
  return UnaryExpresionLazyBlob<LazyBlobCoreFunc, XT>(std::move(x));
}

#define OVERLOAD_UNARY_LAZY_BLOB_OP_FUNC(name, op, ret_type)                          \
  template<typename XType, typename XT = typename std::remove_reference<XType>::type> \
  typename std::enable_if<std::is_base_of<LazyBlobNode, XT>::value,                   \
                          UnaryExpresionLazyBlob<LazyBlobCore##name, XT>>::type       \
  operator op(XType&& x) {                                                            \
    return BuildUnaryLazyBlob<LazyBlobCore##name, XT>(std::move(x));                  \
  }
OF_PP_FOR_EACH_TUPLE(OVERLOAD_UNARY_LAZY_BLOB_OP_FUNC, LAZY_BLOB_UNARY_CORE_OP_FUNC_SEQ);
#undef OVERLOAD_UNARY_LAZY_BLOB_OP_FUNC

//  implementations

template<typename DerivedT>
void LazyBlobIf<DerivedT>::Evaluate1(Blob* output_blob) const {
  using DT = typename DerivedT::dtype;
  CHECK_EQ(shape().NumAxes(), 1);
  DT* dptr = output_blob->mut_dptr<DT>();
  const DerivedT* this_ptr = dynamic_cast<const DerivedT*>(this);
  int64_t dim0_size = shape().At(0);
  FOR_RANGE(int64_t, i, 0, dim0_size) { dptr[i] = this_ptr->At(i); }
}
template<typename DerivedT>
void LazyBlobIf<DerivedT>::Evaluate2(Blob* output_blob) const {
  using DT = typename DerivedT::dtype;
  CHECK_EQ(shape().NumAxes(), 2);
  DT* dptr = output_blob->mut_dptr<DT>();
  const DerivedT* this_ptr = dynamic_cast<const DerivedT*>(this);
  int64_t dim0_size = shape().At(0);
  int64_t dim1_size = shape().At(1);
  int64_t dim0_next_dim_count = shape().Count(1);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    DT* dptr_i = dptr + i * dim0_next_dim_count;
    FOR_RANGE(int64_t, j, 0, dim1_size) { dptr_i[j] = this_ptr->At(i, j); }
  }
}
template<typename DerivedT>
void LazyBlobIf<DerivedT>::Evaluate3(Blob* output_blob) const {
  using DT = typename DerivedT::dtype;
  CHECK_EQ(shape().NumAxes(), 3);
  DT* dptr = output_blob->mut_dptr<DT>();
  const DerivedT* this_ptr = dynamic_cast<const DerivedT*>(this);
  int64_t dim0_size = shape().At(0);
  int64_t dim1_size = shape().At(1);
  int64_t dim2_size = shape().At(2);
  int64_t dim0_next_dim_count = shape().Count(1);
  int64_t dim1_next_dim_count = shape().Count(2);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    DT* dptr_i = dptr + i * dim0_next_dim_count;
    FOR_RANGE(int64_t, j, 0, dim1_size) {
      DT* dptr_j = dptr_i + j * dim1_next_dim_count;
      FOR_RANGE(int64_t, k, 0, dim2_size) { dptr_j[k] = this_ptr->At(i, j, k); }
    }
  }
}
template<typename DerivedT>
void LazyBlobIf<DerivedT>::Evaluate4(Blob* output_blob) const {
  using DT = typename DerivedT::dtype;
  CHECK_EQ(shape().NumAxes(), 4);
  DT* dptr = output_blob->mut_dptr<DT>();
  const DerivedT* this_ptr = dynamic_cast<const DerivedT*>(this);
  int64_t dim0_size = shape().At(0);
  int64_t dim1_size = shape().At(1);
  int64_t dim2_size = shape().At(2);
  int64_t dim3_size = shape().At(3);
  int64_t dim0_next_dim_count = shape().Count(1);
  int64_t dim1_next_dim_count = shape().Count(2);
  int64_t dim2_next_dim_count = shape().Count(3);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    DT* dptr_i = dptr + i * dim0_next_dim_count;
    FOR_RANGE(int64_t, j, 0, dim1_size) {
      DT* dptr_j = dptr_i + j * dim1_next_dim_count;
      FOR_RANGE(int64_t, k, 0, dim2_size) {
        DT* dptr_k = dptr_j + k * dim2_next_dim_count;
        FOR_RANGE(int64_t, s, 0, dim3_size) { dptr_k[s] = this_ptr->At(i, j, k, s); }
      }
    }
  }
}
template<typename DerivedT>
void LazyBlobIf<DerivedT>::Evaluate5(Blob* output_blob) const {
  using DT = typename DerivedT::dtype;
  CHECK_EQ(shape().NumAxes(), 5);
  DT* dptr = output_blob->mut_dptr<DT>();
  const DerivedT* this_ptr = dynamic_cast<const DerivedT*>(this);
  int64_t dim0_size = shape().At(0);
  int64_t dim1_size = shape().At(1);
  int64_t dim2_size = shape().At(2);
  int64_t dim3_size = shape().At(3);
  int64_t dim4_size = shape().At(4);
  int64_t dim0_next_dim_count = shape().Count(1);
  int64_t dim1_next_dim_count = shape().Count(2);
  int64_t dim2_next_dim_count = shape().Count(3);
  int64_t dim3_next_dim_count = shape().Count(4);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    DT* dptr_i = dptr + i * dim0_next_dim_count;
    FOR_RANGE(int64_t, j, 0, dim1_size) {
      DT* dptr_j = dptr_i + j * dim1_next_dim_count;
      FOR_RANGE(int64_t, k, 0, dim2_size) {
        DT* dptr_k = dptr_j + k * dim2_next_dim_count;
        FOR_RANGE(int64_t, s, 0, dim3_size) {
          DT* dptr_s = dptr_k + s * dim3_next_dim_count;
          FOR_RANGE(int64_t, t, 0, dim4_size) { dptr_s[t] = this_ptr->At(i, j, k, s, t); }
        }
      }
    }
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_LAZY_BLOB_H_
