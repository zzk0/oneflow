#include "oneflow/core/persistence/persistent_in_stream_without_local_copy.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/thread/thread_pool.h"

namespace oneflow {

static ThreadPool g_persistent_in_thread_pool(1);

PersistentInStreamWithoutLocalCopy::~PersistentInStreamWithoutLocalCopy() {
  WaitUntilStandByBufferReadyBytesNotEqualZero();
  delete[] standby_buffer_;
  delete[] buffer_;
}

int32_t PersistentInStreamWithoutLocalCopy::ReadLine(std::string* l) {
  if (IsEof()) { return -1; }
  l->clear();
  while (*cur_buf_begin_ != '\n') {
    if (cur_buf_begin_ == cur_buf_end_) {
      UpdateBuffer();
      if (cur_buf_begin_ == cur_buf_end_) {
        return 0;
      } else {
        continue;
      }
    }
    l->push_back(*cur_buf_begin_++);
  }
  ++cur_buf_begin_;
  return 0;
}

int32_t PersistentInStreamWithoutLocalCopy::Read(char* s, size_t n) {
  if (IsEof()) { return -1; }
  while (n) {
    if (cur_buf_begin_ == cur_buf_end_) { UpdateBuffer(); }
    CHECK_LT(cur_buf_begin_, cur_buf_end_);
    size_t copy_size = std::min<size_t>(cur_buf_end_ - cur_buf_begin_, n);
    memcpy(s, cur_buf_begin_, copy_size);
    s += copy_size;
    cur_buf_begin_ += copy_size;
    n -= copy_size;
  }
  return 0;
}

PersistentInStreamWithoutLocalCopy::PersistentInStreamWithoutLocalCopy(fs::FileSystem* fs,
                                                                       const std::string& file_path,
                                                                       uint64_t offset) {
  fs->NewRandomAccessFile(file_path, &file_);
  file_size_ = fs->GetFileSize(file_path);
  CHECK_LT(offset, file_size_);
  standby_buffer_ = new char[Global<JobDesc>::Get()->persistence_buf_byte() + 1];
  standby_buffer_ready_bytes_ = 0;
  cur_file_pos_ = offset;
  file_read_done_ = false;
  buffer_ = new char[Global<JobDesc>::Get()->persistence_buf_byte() + 1];
  cur_buf_begin_ = buffer_;
  cur_buf_end_ = buffer_;
  *cur_buf_end_ = '\0';
  AsyncUpdateStandByBuffer();
}

void PersistentInStreamWithoutLocalCopy::UpdateBuffer() {
  CHECK_EQ(cur_buf_begin_, cur_buf_end_);
  WaitUntilStandByBufferReadyBytesNotEqualZero();
  if (standby_buffer_ready_bytes_ == -1) { return; }
  std::swap(standby_buffer_, buffer_);
  cur_buf_begin_ = buffer_;
  cur_buf_end_ = buffer_ + standby_buffer_ready_bytes_;
  *cur_buf_end_ = '\0';
  standby_buffer_ready_bytes_ = 0;
  AsyncUpdateStandByBuffer();
}

void PersistentInStreamWithoutLocalCopy::WaitUntilStandByBufferReadyBytesNotEqualZero() {
  std::unique_lock<std::mutex> lck(standby_buffer_ready_mtx_);
  standby_buffer_ready_cond_.wait(lck, [this]() { return standby_buffer_ready_bytes_ != 0; });
}

void PersistentInStreamWithoutLocalCopy::AsyncUpdateStandByBuffer() {
  g_persistent_in_thread_pool.AddWork([this]() {
    uint64_t n =
        std::min(Global<JobDesc>::Get()->persistence_buf_byte(), file_size_ - cur_file_pos_);
    if (n > 0) {
      file_->Read(cur_file_pos_, n, standby_buffer_);
      AddNForCurFilePos(n);
    }
    if (cur_file_pos_ == file_size_) { file_read_done_ = true; }
    std::unique_lock<std::mutex> lck(standby_buffer_ready_mtx_);
    if (n > 0) {
      standby_buffer_ready_bytes_ = n;
    } else {
      standby_buffer_ready_bytes_ = -1;
    }
    standby_buffer_ready_cond_.notify_all();
  });
}

bool PersistentInStreamWithoutLocalCopy::IsEof() const {
  return cur_buf_begin_ == cur_buf_end_ && file_read_done_;
}

}  // namespace oneflow
