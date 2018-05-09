#ifndef ONEFLOW_CORE_COMMON_SPIN_CHANNEL_H_
#define ONEFLOW_CORE_COMMON_SPIN_CHANNEL_H_

#include "oneflow/core/common/channel.h"
#include "oneflow/core/common/spin_mutex.h"

namespace oneflow {

template<typename T>
class SpinChannel final : public Channel<T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SpinChannel);
  SpinChannel() = default;
  ~SpinChannel() = default;

  int Send(const T& item) override {
    std::unique_lock<SpinMutex> lock(mutex_);
    return this->SendWithoutLock(item);
  }
  int Receive(T* item) override {
    mutex_.wait_and_lock(std::bind(&SpinChannel<T>::IsReceiveAble, this));
    int ret = this->ReceiveWithoutLock(item);
    mutex_.unlock();
    return ret;
  }
  void CloseSendEnd() override {
    std::unique_lock<SpinMutex> lock(mutex_);
    this->CloseSendEndWithoutLock();
  }
  void CloseReceiveEnd() override {
    std::unique_lock<SpinMutex> lock(mutex_);
    this->CloseReceiveEndWithoutLock();
  }

 private:
  SpinMutex mutex_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SPIN_CHANNEL_H_
