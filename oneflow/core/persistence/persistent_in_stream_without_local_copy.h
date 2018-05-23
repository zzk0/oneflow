#ifndef ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_WITHOUT_LOCAL_COPY_H_
#define ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_WITHOUT_LOCAL_COPY_H_

#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

class PersistentInStreamWithoutLocalCopy : public PersistentInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentInStreamWithoutLocalCopy);
  PersistentInStreamWithoutLocalCopy() = delete;
  virtual ~PersistentInStreamWithoutLocalCopy();

  int32_t ReadLine(std::string* l) override;
  int32_t Read(char* s, size_t n) override;

 protected:
  PersistentInStreamWithoutLocalCopy(fs::FileSystem*, const std::string& file_path,
                                     uint64_t offset);
  virtual void UpdateBuffer();
  virtual void AddNForCurFilePos(uint64_t n) = 0;
  uint64_t file_size() const { return file_size_; }
  char* mut_buffer() { return buffer_; }
  uint64_t cur_file_pos() const { return cur_file_pos_; }
  void set_cur_file_pos(uint64_t val) { cur_file_pos_ = val; }
  void set_cur_buf_begin(char* val) { cur_buf_begin_ = val; }
  void WaitUntilStandByBufferReadyBytesNotEqualZero();

 private:
  void AsyncUpdateStandByBuffer();
  bool IsEof() const;

  std::unique_ptr<fs::RandomAccessFile> file_;
  uint64_t file_size_;

  char* standby_buffer_;
  int64_t standby_buffer_ready_bytes_;
  std::mutex standby_buffer_ready_mtx_;
  std::condition_variable standby_buffer_ready_cond_;
  uint64_t cur_file_pos_;
  std::atomic<bool> file_read_done_;

  char* buffer_;
  char* cur_buf_begin_;
  char* cur_buf_end_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_WITHOUT_LOCAL_COPY_H_
