#include "oneflow/core/auto_placement/df_func.h"

namespace oneflow {

namespace df {

Tensor IndexReduce(const Tensor& input,
                   const std::vector<std::vector<int64_t>>& reduce_indexes) {
  Tensor new_shape_tensor = Reshape(input, Shape({1, input.shape().Count(0)}));
  return ColIndexReduce(new_shape_tensor, reduce_indexes);
}

Tensor ColIndexReduce(const Tensor& input,
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
  return Tensor(out, [=](const Buffer& out_diff) {
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

Tensor Update(Tensor* var, double lr) {
  auto buffer = var->mut_buffer_ptr();
  CHECK(lr > 0);
  double fixed_avg = 1;
  return Tensor(*var, [=](const Buffer& diff) {
    CHECK(buffer->Size() == diff.Size());
    double sum = 0;
    FOR_RANGE(int, i, 0, buffer->Size()) {
      double& w = buffer->At(i);
      double d = diff.At(i);
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

Tensor Reshape(const Tensor& input, const Shape& shape) {
  CHECK(input.shape().Count(0) == shape.Count(0));
  std::shared_ptr<Buffer> out(new Buffer(shape, input.buffer().data()));
  return Tensor(out, [=](const Buffer& out_diff) {
    Buffer input_diff(shape, out_diff.data());
    input.HandleDiff(input_diff);
  });
}

std::vector<Tensor> Clone(const Tensor& input, size_t n) {
  std::shared_ptr<size_t> handled_diff_cnt(new size_t(0));
  std::shared_ptr<Buffer> acc(new Buffer(input.buffer().shape(), 0));
  std::vector<Tensor> out;
  FOR_RANGE(int, i, 0, n) {
    out.emplace_back(input.buffer_ptr(), [=](const Buffer& out_diff) {
      FOR_RANGE(int, i, 0, acc->Size()) {
        acc->mut_data()->at(i) += out_diff.data().at(i);
      }
      ++(*handled_diff_cnt);
      if (*handled_diff_cnt == n) { input.HandleDiff(*acc); }
    });
  }
  return out;
}

Tensor Minus(const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) { out->mut_data()->at(i) *= -1; }
  return Tensor(out, [=](const Buffer& out_diff) {
    Buffer input_diff(out_diff);
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      input_diff.mut_data()->at(i) *= -1;
    }
    input.HandleDiff(input_diff);
  });
}

Tensor Relu(const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) {
    if (input.At(i) < 0) { out->At(i) = 0; }
  }
  return Tensor(out, [=](const Buffer& out_diff) {
    Buffer input_diff(out_diff);
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      if (input.At(i) < 0) { input_diff.At(i) = 0; }
    }
    input.HandleDiff(input_diff);
  });
}

Tensor Abs(const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) {
    double& x = out->mut_data()->at(i);
    x = (x > 0) ? x : -x;
  }
  return Tensor(out, [=](const Buffer& out_diff) {
    Buffer input_diff(out_diff);
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      double& diff = input_diff.mut_data()->at(i);
      diff *= (input.buffer().data().at(i) > 0) ? 1 : -1;
    }
    input.HandleDiff(input_diff);
  });
}

Tensor Tee(const Tensor& input, Tensor* out) {
  *out = input;
  return Tensor(input);
}

Tensor Exp(const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) {
    double& x = out->At(i);
    x = std::exp(x);
  }
  return Tensor(out, [=](const Buffer& out_diff) {
    Buffer input_diff(out_diff);
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      double& diff = input_diff.At(i);
      diff *= out->At(i);
    }
    input.HandleDiff(input_diff);
  });
}

Tensor Tanh(const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) { out->At(i) = std::tanh(input.At(i)); }
  return Tensor(out, [=](const Buffer& out_diff) {
    Buffer input_diff(out_diff);
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      double o = out->At(i);
      input_diff.At(i) *= 1 - o * o;
    }
    input.HandleDiff(input_diff);
  });
}

Tensor Add(const Tensor& a, const Tensor& b) {
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
  return Tensor(out, [=](const Buffer& out_diff) {
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

Tensor Max(const Tensor& a, const Tensor& b) {
  CHECK(a.shape().dim_vec().size() == b.shape().dim_vec().size());
  FOR_RANGE(int, i, 0, a.shape().dim_vec().size()) {
    CHECK(a.shape().dim_vec().at(i) == b.shape().dim_vec().at(i));
  }
  std::shared_ptr<Buffer> out(new Buffer(a.buffer()));
  FOR_RANGE(size_t, i, 0, out->Size()) {
    out->At(i) = std::max(a.At(i), b.At(i));
  }
  return Tensor(out, [=](const Buffer& out_diff) {
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
    a.HandleDiff(a_diff);
    b.HandleDiff(b_diff);
  });
}

Tensor Min(const Tensor& a, const Tensor& b) {
  CHECK(a.shape().dim_vec().size() == b.shape().dim_vec().size());
  FOR_RANGE(int, i, 0, a.shape().dim_vec().size()) {
    CHECK(a.shape().dim_vec().at(i) == b.shape().dim_vec().at(i));
  }
  std::shared_ptr<Buffer> out(new Buffer(a.buffer()));
  FOR_RANGE(size_t, i, 0, out->Size()) {
    out->At(i) = std::min(a.At(i), b.At(i));
  }
  return Tensor(out, [=](const Buffer& out_diff) {
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

Tensor FixedExpectation(const Tensor& input, double e) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  double sum = 0;
  FOR_RANGE(int, i, 0, input.Size()) { sum += input.buffer().data().at(i); }
  double avg = sum / input.Size();
  FOR_RANGE(int, i, 0, out->Size()) { out->mut_data()->at(i) += e - avg; }
  return Tensor(out,
                [=](const Buffer& out_diff) { input.HandleDiff(out_diff); });
}

Tensor MaxElem(const Tensor& input) {
  double max_value = std::numeric_limits<double>::min();
  size_t max_index = 0;
  FOR_RANGE(int, i, 0, input.Size()) {
    if (input.buffer().data().at(i) > max_value) {
      max_value = input.buffer().data().at(i);
      max_index = i;
    }
  }
  std::shared_ptr<Buffer> out(new Buffer(Shape({1}), max_value));
  return Tensor(out, [=](const Buffer& out_diff) {
    Buffer input_diff(input.shape(), 0);
    input_diff.mut_data()->at(max_index) = out_diff.data().at(0);
    input.HandleDiff(input_diff);
  });
}

Tensor MinElem(const Tensor& input) {
  double min_value = std::numeric_limits<double>::max();
  size_t min_index = 0;
  FOR_RANGE(int, i, 0, input.Size()) {
    if (input.At(i) < min_value) {
      min_value = input.buffer().At(i);
      min_index = i;
    }
  }
  std::shared_ptr<Buffer> out(new Buffer(Shape({1}), min_value));
  return Tensor(out, [=](const Buffer& out_diff) {
    Buffer input_diff(input.shape(), 0);
    input_diff.At(min_index) = out_diff.At(0);
    input.HandleDiff(input_diff);
  });
}

Tensor Variance(const Tensor& input) {
  auto copies = Clone(input, 2);
  return Avg(Square(Sub(copies.at(0), Avg(copies.at(1)))));
}

Tensor GeAvg(const Tensor& input) {
  std::vector<std::vector<int64_t>> ge_avg;
  double sum = 0;
  FOR_RANGE(int64_t, i, 0, input.Size()) { sum += input.At(i); }
  double epsilon = 0.000000001;
  double avg = sum / input.Size() - epsilon;
  FOR_RANGE(int64_t, i, 0, input.Size()) {
    if (input.At(i) > avg) { ge_avg.push_back(std::vector<int64_t>{i}); }
  }
  CHECK_GT(ge_avg.size(), 0);
  return IndexReduce(input, ge_avg);
}

Tensor LeAvg(const Tensor& input) {
  std::vector<std::vector<int64_t>> le_avg;
  double sum = 0;
  FOR_RANGE(int64_t, i, 0, input.Size()) { sum += input.At(i); }
  double epsilon = 0.000000001;
  double avg = sum / input.Size() + epsilon;
  FOR_RANGE(int64_t, i, 0, input.Size()) {
    if (input.At(i) < avg) { le_avg.push_back(std::vector<int64_t>{i}); }
  }
  CHECK_GT(le_avg.size(), 0);
  return IndexReduce(input, le_avg);
}

Tensor DoubleVariance(const Tensor& input) {
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
  return Add(Variance(IndexReduce(input_copies.at(0), ge_avg)),
             Variance(IndexReduce(input_copies.at(1), le_avg)));
}

Tensor AvgAbsDeviation(const Tensor& input) {
  auto copies = Clone(input, 2);
  return Avg(Abs(Sub(copies.at(0), Avg(copies.at(1)))));
}

Tensor Sum(const Tensor& input) {
  double sum = 0;
  FOR_RANGE(int, i, 0, input.Size()) { sum += input.At(i); }
  std::shared_ptr<Buffer> out(new Buffer(Shape({1}), sum));
  return Tensor(out, [=](const Buffer& out_diff) {
    Buffer input_diff(input.shape(), 0);
    double diff = out_diff.data().at(0);
    FOR_RANGE(int, i, 0, input_diff.Size()) { input_diff.At(i) = diff; }
    input.HandleDiff(input_diff);
  });
}

Tensor Avg(const Tensor& input) {
  CHECK(input.Size() > 0);
  Tensor sum = Sum(input);
  double avg = sum.At(0) / input.Size();
  std::shared_ptr<Buffer> out(new Buffer(Shape({1}), avg));
  return Tensor(out, [=](const Buffer& out_diff) { sum.HandleDiff(out_diff); });
}

Tensor Mul(const Tensor& a, const Tensor& b) {
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
  return Tensor(out, [=](const Buffer& out_diff) {
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

Tensor ElemWiseMul(const Tensor& a, const Tensor& b) {
  CHECK(a.Size() == b.Size());
  std::shared_ptr<Buffer> out(new Buffer(a.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) { out->At(i) *= b.At(i); }
  return Tensor(out, [=](const Buffer& out_diff) {
    Buffer a_diff(out_diff);
    FOR_RANGE(int, i, 0, a_diff.Size()) { a_diff.At(i) *= b.At(i); }
    a.HandleDiff(a_diff);
    Buffer b_diff(out_diff);
    FOR_RANGE(int, i, 0, b_diff.Size()) { b_diff.At(i) *= a.At(i); }
    b.HandleDiff(b_diff);
  });
}

Tensor Reciprocal(const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  FOR_RANGE(int, i, 0, out->Size()) { out->At(i) = 1 / out->At(i); }
  return Tensor(out, [=](const Buffer& out_diff) {
    Buffer input_diff(out_diff);
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      double o = out->At(i);
      input_diff.At(i) *= -1 * o * o;
    }
    input.HandleDiff(input_diff);
  });
}

Tensor ElemWiseDiv(const Tensor& a, const Tensor& b) {
  return ElemWiseMul(a, Reciprocal(b));
}

Tensor MatrixRowSum(const Tensor& input) {
  CHECK(input.shape().dim_vec().size() == 2);
  std::shared_ptr<Buffer> out(new Buffer(Shape({input.shape().At(0)}), 0));
  FOR_RANGE(int, i, 0, input.shape().At(0)) {
    FOR_RANGE(int, j, 0, input.shape().At(1)) { out->At(i) += input.At(i, j); }
  }
  return Tensor(out, [=](const Buffer& out_diff) {
    Buffer input_diff(input.buffer());
    FOR_RANGE(int, i, 0, input_diff.shape().At(0)) {
      FOR_RANGE(int, j, 0, input_diff.shape().At(1)) {
        input_diff.At(i, j) = out_diff.At(i);
      }
    }
    input.HandleDiff(input_diff);
  });
}

Tensor TensorProduct(const Tensor& a, const Tensor& b) {
  std::vector<int64_t> dim_vec(a.shape().dim_vec());
  for (int64_t d : b.shape().dim_vec()) { dim_vec.push_back(d); }
  std::shared_ptr<Buffer> out(new Buffer(Shape(dim_vec), 1));
  FOR_RANGE(int, i, 0, a.Size()) {
    FOR_RANGE(int, j, 0, b.Size()) {
      out->At(i * b.Size() + j) = a.At(i) * b.At(j);
    }
  }
  return Tensor(out, [=](const Buffer& out_diff) {
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

Tensor MatrixColSum(const Tensor& input) {
  CHECK(input.shape().dim_vec().size() == 2);
  std::shared_ptr<Buffer> out(new Buffer(Shape({input.shape().At(1)}), 0));
  FOR_RANGE(int, i, 0, input.shape().At(0)) {
    FOR_RANGE(int, j, 0, input.shape().At(1)) { out->At(j) += input.At(i, j); }
  }
  return Tensor(out, [=](const Buffer& out_diff) {
    Buffer input_diff(input.buffer());
    FOR_RANGE(int, i, 0, input_diff.shape().At(0)) {
      FOR_RANGE(int, j, 0, input_diff.shape().At(1)) {
        input_diff.At(i, j) = out_diff.At(j);
      }
    }
    input.HandleDiff(input_diff);
  });
}

Tensor MatrixColMax(const Tensor& input) {
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
  return Tensor(out, [=](const Buffer& out_diff) {
    CHECK(out_diff.Size() == input.shape().At(1));
    Buffer input_diff(input.shape(), 0);
    FOR_RANGE(int, j, 0, out_diff.Size()) {
      input_diff.At(max_index->at(j), j) = out_diff.At(j);
    }
    input.HandleDiff(input_diff);
  });
}

Tensor Sub(const Tensor& a, const Tensor& b) { return Add(a, Minus(b)); }

Tensor Square(const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  for (double& x : *out->mut_data()) { x *= x; }
  return Tensor(out, [input](const Buffer& out_diff) {
    Buffer input_diff(input.buffer());
    FOR_RANGE(int, i, 0, input_diff.data().size()) {
      double& id = input_diff.mut_data()->at(i);
      double od = out_diff.data().at(i);
      id = 2 * id * od;
    }
    input.HandleDiff(input_diff);
  });
}

Tensor Sqrt(const Tensor& input) {
  std::shared_ptr<Buffer> out(new Buffer(input.buffer()));
  for (double& x : *out->mut_data()) { x = std::sqrt(x); }
  return Tensor(out, [input](const Buffer& out_diff) {
    Buffer input_diff(input.buffer());
    FOR_RANGE(int, i, 0, input_diff.Size()) {
      double& id = input_diff.At(i);
      double od = out_diff.At(i);
      id *= 0.5 / od;
    }
    input.HandleDiff(input_diff);
  });
}

Tensor StandardDeviation(const Tensor& a) { return Sqrt(Variance(a)); }

Tensor Backward(const Tensor& loss) {
  CHECK(loss.buffer().data().size() == 1);
  Buffer diff(Shape({1}), 1);
  loss.HandleDiff(diff);
  std::shared_ptr<Buffer> out(new Buffer(loss.buffer()));
  return Tensor(out, [](const Buffer&) {});
}

}  // namespace df

}  // namespace oneflow
