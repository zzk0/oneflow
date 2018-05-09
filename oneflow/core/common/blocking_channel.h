#ifndef ONEFLOW_CORE_COMMON_BLOCKING_CHANNEL_H_
#define ONEFLOW_CORE_COMMON_BLOCKING_CHANNEL_H_

#include "oneflow/core/common/channel.h"

namespace oneflow {

template<typename T>
class BlockingChannel final : public Channel<T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlockingChannel);
  BlockingChannel() = default;
  ~BlockingChannel() = default;

  int Send(const T& item) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int ret = this->SendWithoutLock(item);
    if (ret == 0) { cv_.notify_one(); }
    return ret;
  }
  int Receive(T* item) override {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, std::bind(&BlockingChannel<T>::IsReceiveAble, this));
    return this->ReceiveWithoutLock(item);
  }
  void CloseSendEnd() override {
    std::unique_lock<std::mutex> lock(mutex_);
    this->CloseSendEndWithoutLock();
    cv_.notify_all();
  }
  void CloseReceiveEnd() override {
    std::unique_lock<std::mutex> lock(mutex_);
    this->CloseReceiveEndWithoutLock();
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_BLOCKING_CHANNEL_H_
