#ifndef ONEFLOW_CORE_COMMON_SPIN_MUTEX_H_
#define ONEFLOW_CORE_COMMON_SPIN_MUTEX_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class SpinMutex final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SpinMutex);
  SpinMutex() : flag_(ATOMIC_FLAG_INIT) {}
  ~SpinMutex() = default;

  void lock() {
    while (flag_.test_and_set()) {}
  }

  void wait_and_lock(std::function<bool()> cond) {
    while (true) {
      if (flag_.test_and_set()) {
        if (cond()) {
          return;
        } else {
          flag_.clear();
        }
      }
    }
  }

  void unlock() { flag_.clear(); }

 private:
  std::atomic_flag flag_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SPIN_MUTEX_H_
