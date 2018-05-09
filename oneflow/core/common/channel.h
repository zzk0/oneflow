#ifndef ONEFLOW_CORE_COMMON_CHANNEL_H_
#define ONEFLOW_CORE_COMMON_CHANNEL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T>
class Channel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Channel);
  Channel() : is_send_closed_(false), is_receive_closed_(false) {}
  virtual ~Channel() = default;

  // return code
  //   0 : success send item
  //  -1 : fail (send end has been closed)
  virtual int Send(const T& item) = 0;

  //  If the channel is empty, the thread calling Receive() would be blocked.
  //  return value
  //    0 : success -- if successfully get the item ref in val_
  //   -1 : fail    -- when the channel tell the owner thread should exit
  virtual int Receive(T* item) = 0;

  // close the channel's send end, the thread can't send item to the channel
  virtual void CloseSendEnd() = 0;

  // close the channel's receive end , the thread can't receive item from channel
  virtual void CloseReceiveEnd() = 0;

 protected:
  bool IsReceiveAble() { return !val_.empty() || is_receive_closed_ || is_send_closed_; }

  int SendWithoutLock(const T& item) {
    if (is_send_closed_) { return -1; }
    val_.push(item);
    return 0;
  }

  int ReceiveWithoutLock(T* item) {
    if (val_.empty() || is_receive_closed_) { return -1; }
    *item = val_.front();
    val_.pop();
    return 0;
  }

  void CloseSendEndWithoutLock() { is_send_closed_ = true; }

  void CloseReceiveEndWithoutLock() { is_receive_closed_ = true; }

 private:
  std::queue<T> val_;
  bool is_send_closed_;
  bool is_receive_closed_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CHANNEL_H_
