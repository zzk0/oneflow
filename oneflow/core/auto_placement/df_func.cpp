#include "oneflow/core/auto_placement/df_func.h"

#define DEFINE_RUN_TIME_CNT_BOX() \
  std::shared_ptr<size_t> __run_time_cnt_box__(new size_t(-1))

#define RUN_ONLY_ONE_TIME() CHECK_LT(++(*__run_time_cnt_box__), 1)

#define SET_FW_CALLER() auto fw_caller = caller;

namespace oneflow {

namespace df {

Tensor _IndexReduce(const std::string& caller, const Tensor& input,
                    const std::vector<std::vector<int64_t>>& reduce_indexes) {
  Tensor new_shape_tensor = Reshape(input, Shape({1, input.shape().Count(0)}));
  return ColIndexReduce(new_shape_tensor, reduce_indexes);
}

Tensor _ColIndexReduce(
    const std::string& caller, const Tensor& input,
    const std::vector<std::vector<int64_t>>& reduce_indexes) {
  CHECK(input.shape().dim_vec().size() == 2);
  auto shape =
      Shape({input.shape().At(0), static_cast<int64_t>(reduce_indexes.size())});
  std::shared_ptr<Buffer> out(new Buffer(shape, 0));
  FOR_RANGE(int, i, 0, out->shape().At(0)) {
    FOR_RANGE(int, j, 0, out->shape().At(1)) {
      for (int64_t index : reduce_indexes.at(j)) {
        out->At(i, j) += input.At(i, index);
      }
    }
  }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(input.shape(), 0);
    FOR_RANGE(int, i, 0, out_diff.shape().At(0)) {
      FOR_RANGE(int, j, 0, out_diff.shape().At(1)) {
        for (int64_t index : reduce_indexes.at(j)) {
          input_diff.At(i, index) += out_diff.At(i, j);
        }
      }
    }
    input.HandleDiff(input_diff);
  });
}

Tensor _Update(const std::string& caller, Tensor* var, double lr) {
  auto buffer = var->mut_buffer_ptr();
  CHECK(lr > 0);
  double fixed_avg = 1;

  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(*var, [=](const Buffer& diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    CHECK(buffer->Size() == diff.Size());
    double max_diff = 0;
    FOR_RANGE(int, i, 0, diff.Size()) {
      max_diff = std::max(max_diff, std::abs(diff.At(i)));
    }
    double sum = 0;
    FOR_RANGE(int, i, 0, buffer->Size()) {
      double& w = buffer->At(i);
      double d = diff.At(i);
      if (max_diff > 10) { d *= 10 / max_diff; }
      w -= lr * d;
      sum += w;
    }
    double avg = sum / diff.Size();
    FOR_RANGE(int, i, 0, buffer->Size()) {
      double& w = buffer->At(i);
      w += fixed_avg - avg;
      if (w < 0) { w *= -0.5; }
    }
  });
}

Tensor _Reshape(const std::string& caller, const Tensor& input,
                const Shape& shape) {
  CHECK(input.shape().Count(0) == shape.Count(0));
  std::shared_ptr<Buffer> out(new Buffer(shape, input.buffer().data()));
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(shape, out_diff.data());
    input.HandleDiff(input_diff);
  });
}

Tensor _DiffWatch(const std::string& caller, const Tensor& input,
                  const std::function<void(const Buffer& out_diff)>& Handler) {
  std::shared_ptr<Buffer> out = input.buffer_ptr();
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    // RUN_ONLY_ONE_TIME();
    Handler(out_diff);
    input.HandleDiff(out_diff);
  });
}

std::vector<Tensor> _Clone(const std::string& caller, const Tensor& input,
                           size_t n) {
  std::shared_ptr<size_t> handled_diff_cnt(new size_t(0));
  std::shared_ptr<Buffer> acc(new Buffer(input.buffer().shape(), 0));
  std::vector<Tensor> out;
  FOR_RANGE(int, i, 0, n) {
    out.emplace_back(input.buffer_ptr(), [=](const Buffer& out_diff) {
      SET_FW_CALLER();
      FOR_RANGE(int, i, 0, acc->Size()) {
        acc->mut_data()->at(i) += out_diff.data().at(i);
      }
      ++(*handled_diff_cnt);
      if (*handled_diff_cnt == n) { input.HandleDiff(*acc); }
      CHECK_LE(*handled_diff_cnt, n);
    });
  }
  return out;
}

Tensor _Minus(const std::string& caller, const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) { out->mut_data()->at(i) *= -1; }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(out_diff);
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      input_diff.mut_data()->at(i) *= -1;
    }
    input.HandleDiff(input_diff);
  });
}

Tensor _Relu(const std::string& caller, const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) {
    if (input.At(i) < 0) { out->At(i) = 0; }
  }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(out_diff);
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      if (input.At(i) < 0) { input_diff.At(i) = 0; }
    }
    input.HandleDiff(input_diff);
  });
}

Tensor _Abs(const std::string& caller, const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) {
    double& x = out->mut_data()->at(i);
    x = (x > 0) ? x : -x;
  }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(out_diff);
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      double& diff = input_diff.mut_data()->at(i);
      diff *= (input.buffer().data().at(i) > 0) ? 1 : -1;
    }
    input.HandleDiff(input_diff);
  });
}

Tensor _Tee(const std::string& caller, const Tensor& input, Tensor* out) {
  *out = input;
  return Tensor(input);
}

Tensor _Exp(const std::string& caller, const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) {
    double& x = out->At(i);
    x = std::exp(x);
  }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(out_diff);
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      double& diff = input_diff.At(i);
      diff *= out->At(i);
    }
    input.HandleDiff(input_diff);
  });
}

Tensor _Tanh(const std::string& caller, const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) { out->At(i) = std::tanh(input.At(i)); }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(out_diff);
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      double o = out->At(i);
      input_diff.At(i) *= 1 - o * o;
    }
    input.HandleDiff(input_diff);
  });
}

Tensor _Add(const std::string& caller, const Tensor& a, const Tensor& b) {
  Tensor big = a;
  Tensor small = b;
  if (a.Size() < b.Size()) {
    big = b;
    small = a;
  }
  CHECK(big.Size() % small.Size() == 0);
  std::shared_ptr<Buffer> out(new Buffer(big.buffer()));
  size_t small_size = small.Size();
  size_t group_size = big.Size() / small_size;
  FOR_RANGE(int, i, 0, small_size) {
    FOR_RANGE(int, j, 0, group_size) {
      out->At(i * group_size + j) += small.At(i);
    }
  }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    big.HandleDiff(out_diff);
    Buffer small_diff(small.shape(), 0);
    FOR_RANGE(int, i, 0, small_size) {
      FOR_RANGE(int, j, 0, group_size) {
        small_diff.At(i) += out_diff.At(i * group_size + j);
      }
    }
    small.HandleDiff(small_diff);
  });
}

Tensor _Max(const std::string& caller, const Tensor& a, const Tensor& b) {
  CHECK(a.shape().dim_vec().size() == b.shape().dim_vec().size());
  FOR_RANGE(int, i, 0, a.shape().dim_vec().size()) {
    CHECK(a.shape().dim_vec().at(i) == b.shape().dim_vec().at(i));
  }
  std::shared_ptr<Buffer> out(new Buffer(a.buffer()));
  FOR_RANGE(size_t, i, 0, out->Size()) {
    out->At(i) = std::max(a.At(i), b.At(i));
  }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer a_diff(out_diff.shape(), 0);
    Buffer b_diff(out_diff.shape(), 0);
    FOR_RANGE(size_t, i, 0, out_diff.Size()) {
      if (a.At(i) >= b.At(i)) {
        a_diff.At(i) = out_diff.At(i);
        b_diff.At(i) = 0;
      }
      if (b.At(i) >= a.At(i)) {
        b_diff.At(i) = out_diff.At(i);
        a_diff.At(i) = 0;
      }
    }
    b.HandleDiff(b_diff);
    a.HandleDiff(a_diff);
  });
}

Tensor _Min(const std::string& caller, const Tensor& a, const Tensor& b) {
  CHECK(a.shape().dim_vec().size() == b.shape().dim_vec().size());
  FOR_RANGE(int, i, 0, a.shape().dim_vec().size()) {
    CHECK(a.shape().dim_vec().at(i) == b.shape().dim_vec().at(i));
  }
  std::shared_ptr<Buffer> out(new Buffer(a.buffer()));
  FOR_RANGE(size_t, i, 0, out->Size()) {
    out->At(i) = std::min(a.At(i), b.At(i));
  }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer a_diff(out_diff.shape(), 0);
    Buffer b_diff(out_diff.shape(), 0);
    FOR_RANGE(size_t, i, 0, out_diff.Size()) {
      if (a.At(i) <= b.At(i)) {
        a_diff.At(i) = out_diff.At(i);
        b_diff.At(i) = 0;
      }
      if (b.At(i) <= a.At(i)) {
        b_diff.At(i) = out_diff.At(i);
        a_diff.At(i) = 0;
      }
    }
    a.HandleDiff(a_diff);
    b.HandleDiff(b_diff);
  });
}

Tensor _FixedMaxVal(const std::string& caller, const Tensor& input, double e) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  double max_val = std::numeric_limits<double>::min();
  FOR_RANGE(int, i, 0, input.Size()) {
    max_val = std::max(max_val, input.At(i));
  }
  FOR_RANGE(int, i, 0, out->Size()) { out->mut_data()->at(i) += e - max_val; }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    input.HandleDiff(out_diff);
  });
}

Tensor _FixedExpectation(const std::string& caller, const Tensor& input,
                         double e) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  double sum = 0;
  FOR_RANGE(int, i, 0, input.Size()) { sum += input.buffer().data().at(i); }
  double avg = sum / input.Size();
  FOR_RANGE(int, i, 0, out->Size()) { out->mut_data()->at(i) += e - avg; }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    input.HandleDiff(out_diff);
  });
}

Tensor _MaxElem(const std::string& caller, const Tensor& input) {
  double max_value = std::numeric_limits<double>::min();
  size_t max_index = 0;
  FOR_RANGE(int, i, 0, input.Size()) {
    if (input.buffer().data().at(i) > max_value) {
      max_value = input.buffer().data().at(i);
      max_index = i;
    }
  }
  std::shared_ptr<Buffer> out(new Buffer(Shape({1}), max_value));
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(input.shape(), 0);
    input_diff.mut_data()->at(max_index) = out_diff.data().at(0);
    input.HandleDiff(input_diff);
  });
}

Tensor _MinElem(const std::string& caller, const Tensor& input) {
  double min_value = std::numeric_limits<double>::max();
  size_t min_index = 0;
  FOR_RANGE(int, i, 0, input.Size()) {
    if (input.At(i) < min_value) {
      min_value = input.buffer().At(i);
      min_index = i;
    }
  }
  std::shared_ptr<Buffer> out(new Buffer(Shape({1}), min_value));
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(input.shape(), 0);
    input_diff.At(min_index) = out_diff.At(0);
    input.HandleDiff(input_diff);
  });
}

Tensor _Variance(const std::string& caller, const Tensor& input) {
  auto copies = Clone(input, 2);
  return Avg(Square(Sub(copies.at(0), Avg(copies.at(1)))));
}

Tensor _GeMean(const std::string& caller, const Tensor& input) {
  std::vector<std::vector<int64_t>> ge_avg;
  double sum = 0;
  FOR_RANGE(int64_t, i, 0, input.Size()) { sum += input.At(i); }
  double avg = sum / input.Size();
  double epsilon = 0.000000009;
  FOR_RANGE(int64_t, i, 0, input.Size()) {
    if (input.At(i) >= (avg - epsilon)) {
      ge_avg.push_back(std::vector<int64_t>{i});
    }
  }
  CHECK_GT(ge_avg.size(), 0);
  auto input_copies = Clone(input, 2);
  return IndexReduce(input_copies.at(0), ge_avg);
}

Tensor _DoubleAvgAbsDeviation(const std::string& caller, const Tensor& input) {
  std::vector<std::vector<int64_t>> ge_avg;
  std::vector<std::vector<int64_t>> le_avg;
  double sum = 0;
  FOR_RANGE(int64_t, i, 0, input.Size()) { sum += input.At(i); }
  double avg = sum / input.Size();
  double epsilon = 0.000000009;
  FOR_RANGE(int64_t, i, 0, input.Size()) {
    if (input.At(i) >= (avg - epsilon)) {
      ge_avg.push_back(std::vector<int64_t>{i});
    }
    if (input.At(i) <= (avg + epsilon)) {
      le_avg.push_back(std::vector<int64_t>{i});
    }
  }
  CHECK_GT(ge_avg.size(), 0);
  CHECK_GT(le_avg.size(), 0);
  auto input_copies = Clone(input, 2);
  return ADD(AvgAbsDeviation(IndexReduce(input_copies.at(0), ge_avg)),
             AvgAbsDeviation(IndexReduce(input_copies.at(1), le_avg)));
}

Tensor _DoubleVariance(const std::string& caller, const Tensor& input) {
  std::vector<std::vector<int64_t>> ge_avg;
  std::vector<std::vector<int64_t>> le_avg;
  double sum = 0;
  FOR_RANGE(int64_t, i, 0, input.Size()) { sum += input.At(i); }
  double avg = sum / input.Size();
  double epsilon = 0.000000009;
  FOR_RANGE(int64_t, i, 0, input.Size()) {
    if (input.At(i) >= (avg - epsilon)) {
      ge_avg.push_back(std::vector<int64_t>{i});
    }
    if (input.At(i) <= (avg + epsilon)) {
      le_avg.push_back(std::vector<int64_t>{i});
    }
  }
  CHECK_GT(ge_avg.size(), 0);
  CHECK_GT(le_avg.size(), 0);
  auto input_copies = Clone(input, 2);
  return ADD(Variance(IndexReduce(input_copies.at(0), ge_avg)),
             Variance(IndexReduce(input_copies.at(1), le_avg)));
}

Tensor _AvgAbsDeviation(const std::string& caller, const Tensor& input) {
  auto copies = Clone(input, 2);
  return Avg(Abs(Sub(copies.at(0), Avg(copies.at(1)))));
}

Tensor _Sum(const std::string& caller, const Tensor& input) {
  double sum = 0;
  FOR_RANGE(int, i, 0, input.Size()) { sum += input.At(i); }
  std::shared_ptr<Buffer> out(new Buffer(Shape({1}), sum));
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(input.shape(), 0);
    double diff = out_diff.data().at(0);
    FOR_RANGE(int, i, 0, input_diff.Size()) { input_diff.At(i) = diff; }
    input.HandleDiff(input_diff);
  });
}

Tensor _Avg(const std::string& caller, const Tensor& input) {
  CHECK(input.Size() > 0);
  Tensor sum = Sum(input);
  double avg = sum.At(0) / input.Size();
  std::shared_ptr<Buffer> out(new Buffer(Shape({1}), avg));
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    sum.HandleDiff(out_diff);
  });
}

Tensor _Mul(const std::string& caller, const Tensor& a, const Tensor& b) {
  Tensor big = a;
  Tensor small = b;
  if (a.Size() < b.Size()) {
    big = b;
    small = a;
  }
  CHECK(big.Size() % small.Size() == 0);
  std::shared_ptr<Buffer> out(new Buffer(big.buffer()));
  size_t small_size = small.Size();
  size_t group_size = big.Size() / small_size;
  FOR_RANGE(int, i, 0, small_size) {
    FOR_RANGE(int, j, 0, group_size) {
      out->At(i * group_size + j) *= small.At(i);
    }
  }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer big_diff(out_diff);
    FOR_RANGE(int, i, 0, small_size) {
      FOR_RANGE(int, j, 0, group_size) {
        big_diff.At(i * group_size + j) *= small.At(i);
      }
    }
    big.HandleDiff(big_diff);
    Buffer small_diff(small.shape(), 0);
    FOR_RANGE(int, i, 0, small_size) {
      FOR_RANGE(int, j, 0, group_size) {
        size_t index = i * group_size + j;
        small_diff.At(i) += out_diff.At(index) * big.At(index);
      }
    }
    small.HandleDiff(small_diff);
  });
}

Tensor _ElemWiseMul(const std::string& caller, const Tensor& a,
                    const Tensor& b) {
  CHECK(a.Size() == b.Size());
  std::shared_ptr<Buffer> out(new Buffer(a.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) { out->At(i) *= b.At(i); }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer a_diff(out_diff);
    FOR_RANGE(int, i, 0, a_diff.Size()) { a_diff.At(i) *= b.At(i); }
    a.HandleDiff(a_diff);
    Buffer b_diff(out_diff);
    FOR_RANGE(int, i, 0, b_diff.Size()) { b_diff.At(i) *= a.At(i); }
    b.HandleDiff(b_diff);
  });
}

Tensor _Reciprocal(const std::string& caller, const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) { out->At(i) = 1 / out->At(i); }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(out_diff);
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      double o = out->At(i);
      input_diff.At(i) *= -1 * o * o;
    }
    input.HandleDiff(input_diff);
  });
}

Tensor _ElemWiseDiv(const std::string& caller, const Tensor& a,
                    const Tensor& b) {
  return ElemWiseMul(a, Reciprocal(b));
}

Tensor _MatrixRowSum(const std::string& caller, const Tensor& input) {
  CHECK(input.shape().dim_vec().size() == 2);
  std::shared_ptr<Buffer> out(new Buffer(Shape({input.shape().At(0)}), 0));
  FOR_RANGE(int, i, 0, input.shape().At(0)) {
    FOR_RANGE(int, j, 0, input.shape().At(1)) { out->At(i) += input.At(i, j); }
  }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(input.buffer());
    FOR_RANGE(int, i, 0, input_diff.shape().At(0)) {
      FOR_RANGE(int, j, 0, input_diff.shape().At(1)) {
        input_diff.At(i, j) = out_diff.At(i);
      }
    }
    input.HandleDiff(input_diff);
  });
}

Tensor _TensorProduct(const std::string& caller, const Tensor& a,
                      const Tensor& b) {
  std::vector<int64_t> dim_vec(a.shape().dim_vec());
  for (int64_t d : b.shape().dim_vec()) { dim_vec.push_back(d); }
  std::shared_ptr<Buffer> out(new Buffer(Shape(dim_vec), 1));
  FOR_RANGE(int, i, 0, a.Size()) {
    FOR_RANGE(int, j, 0, b.Size()) {
      out->At(i * b.Size() + j) = a.At(i) * b.At(j);
    }
  }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer a_diff(a.shape(), 0);
    FOR_RANGE(int, i, 0, a.Size()) {
      FOR_RANGE(int, j, 0, b.Size()) {
        a_diff.At(i) += out_diff.At(i * b.Size() + j) * b.At(j);
      }
    }
    a.HandleDiff(a_diff);

    Buffer b_diff(b.shape(), 0);
    FOR_RANGE(int, i, 0, a.Size()) {
      FOR_RANGE(int, j, 0, b.Size()) {
        b_diff.At(j) += out_diff.At(i * b.Size() + j) * a.At(i);
      }
    }
    b.HandleDiff(b_diff);
  });
}

Tensor _MatrixColSum(const std::string& caller, const Tensor& input) {
  CHECK(input.shape().dim_vec().size() == 2);
  std::shared_ptr<Buffer> out(new Buffer(Shape({input.shape().At(1)}), 0));
  FOR_RANGE(int, i, 0, input.shape().At(0)) {
    FOR_RANGE(int, j, 0, input.shape().At(1)) { out->At(j) += input.At(i, j); }
  }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(input.buffer());
    FOR_RANGE(int, i, 0, input_diff.shape().At(0)) {
      FOR_RANGE(int, j, 0, input_diff.shape().At(1)) {
        input_diff.At(i, j) = out_diff.At(j);
      }
    }
    input.HandleDiff(input_diff);
  });
}

Tensor _MatrixColMax(const std::string& caller, const Tensor& input) {
  CHECK(input.shape().dim_vec().size() == 2);
  int64_t out_size = input.shape().At(1);
  std::shared_ptr<Buffer> out(
      new Buffer(Shape({out_size}), std::numeric_limits<double>::min()));
  std::shared_ptr<std::vector<size_t>> max_index(
      new std::vector<size_t>(out_size));
  FOR_RANGE(int, j, 0, input.shape().At(1)) {
    FOR_RANGE(int, i, 0, input.shape().At(0)) {
      if (input.At(i, j) > out->At(j)) {
        out->At(j) = input.At(i, j);
        max_index->at(j) = i;
      }
    }
  }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    CHECK(out_diff.Size() == input.shape().At(1));
    Buffer input_diff(input.shape(), 0);
    FOR_RANGE(int, j, 0, out_diff.Size()) {
      input_diff.At(max_index->at(j), j) = out_diff.At(j);
    }
    input.HandleDiff(input_diff);
  });
}

Tensor _Sub(const std::string& caller, const Tensor& a, const Tensor& b) {
  return ADD(a, Minus(b));
}

Tensor _Square(const std::string& caller, const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  for (double& x : *out->mut_data()) { x *= x; }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(input.buffer());
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      double& id = input_diff.At(i);
      double od = out_diff.At(i);
      id = 2 * id * od;
    }
    input.HandleDiff(input_diff);
  });
}

Tensor _Sqrt(const std::string& caller, const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  for (double& x : *out->mut_data()) { x = std::sqrt(x); }
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer& out_diff) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
    Buffer input_diff(out_diff);
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      double& id = input_diff.At(i);
      double o = out->At(i);
      id *= 0.5 / o;
    }
    std::cout << std::endl;
    input.HandleDiff(input_diff);
  });
}

Tensor _StandardDeviation(const std::string& caller, const Tensor& a) {
  return Sqrt(Variance(a));
}

Tensor _Backward(const std::string& caller, const Tensor& loss) {
  CHECK(loss.buffer().data().size() == 1);
  Buffer diff(Shape({1}), 1);
  loss.HandleDiff(diff);
  std::shared_ptr<Buffer> out(new Buffer(loss.buffer()));
  DEFINE_RUN_TIME_CNT_BOX();
  return Tensor(out, [=](const Buffer&) {
    SET_FW_CALLER();
    RUN_ONLY_ONE_TIME();
  });
}

}  // namespace df

}  // namespace oneflow
