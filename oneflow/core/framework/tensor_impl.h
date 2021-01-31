#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/register/logical_blob_id.cfg.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.cfg.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/framework/blob_register.h"

namespace oneflow {

namespace one {

class TensorImpl {
 public:
  TensorImpl() = default;
  TensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
           const std::shared_ptr<compatible_py::Distribute>& distribute);
  TensorImpl(const TensorImpl& tensor) = default; 
  virtual ~TensorImpl() = default; 

  virtual std::shared_ptr<cfg::LogicalBlobId> lbi() const; 
  virtual std::string logical_blob_name() const;
  virtual std::string op_name() const;
  virtual std::string blob_name() const;
  virtual int64_t batch_axis() const;
  virtual bool has_batch_axis() const;
  virtual std::shared_ptr<compatible_py::Distribute> distribute() const;
  virtual std::string unique_name() const;
  void set_distribute(const std::shared_ptr<compatible_py::Distribute> distribute);


  virtual std::string job_name() const = 0;
  virtual int64_t parallel_size() = 0;
  virtual void set_job_name(std::string job_name) = 0;
 
 protected:
  Maybe<std::string> Distribute2Str() const;
  std::shared_ptr<cfg::LogicalBlobId> lbi_;
  std::shared_ptr<compatible_py::Distribute> distribute_;
  std::string lbn_;
};

class ConsistentTensorImpl : public TensorImpl {
 public:
  ConsistentTensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                 const std::shared_ptr<compatible_py::Distribute>& distribute);
  ConsistentTensorImpl(const ConsistentTensorImpl& impl) = default;
  ~ConsistentTensorImpl() = default;

  std::string job_name() const override;
  int64_t parallel_size() override;
  void set_job_name(std::string job_name) override;

  virtual std::shared_ptr<Shape> shape() const = 0;
  virtual DataType dtype() const = 0;
  virtual int64_t split_axis() const = 0;
  virtual bool is_dynamic() const = 0;
  virtual bool is_tensor_list() const = 0;
  virtual std::shared_ptr<cfg::ParallelConf> parallel_conf() const = 0;
  virtual int64_t numpy_size() const = 0;
  virtual int64_t numpy_list_size() const = 0;
  virtual std::shared_ptr<compatible_py::BlobObject> blob_object() const = 0;

 private:
  std::string job_name_;
  int64_t parallel_size_;
};

class MirroredTensorImpl : public TensorImpl {
 public:
  MirroredTensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
               const std::shared_ptr<compatible_py::Distribute>& distribute);
  virtual ~MirroredTensorImpl() = default;

  std::string job_name() const override;
  int64_t parallel_size() override;
  void set_job_name(std::string job_name) override;

  virtual std::shared_ptr<Shape> shape() const = 0;
  virtual DataType dtype() const = 0;
  virtual int64_t split_axis() const = 0;
  virtual bool is_dynamic() const = 0;
  virtual bool is_tensor_list() const = 0;
  virtual std::shared_ptr<cfg::ParallelConf> parallel_conf() const = 0;
  virtual int64_t numpy_size() const = 0;
  virtual int64_t numpy_list_size() const = 0;
  virtual std::shared_ptr<compatible_py::BlobObject> blob_object() const = 0;

 private:
  std::string job_name_;
  int64_t parallel_size_;
};

class LazyConsistentTensorImpl : public ConsistentTensorImpl {
 public:
  LazyConsistentTensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                     const std::shared_ptr<compatible_py::Distribute>& distribute);
  LazyConsistentTensorImpl(const LazyConsistentTensorImpl& lazy_consistent_blob) = default;
  ~LazyConsistentTensorImpl() = default;

  virtual std::string get_lazy_shape_log_warning() const;
  std::shared_ptr<Shape> shape() const override;
  DataType dtype() const override;
  int64_t batch_axis() const override;
  int64_t split_axis() const override;
  bool is_dynamic() const override;
  bool is_tensor_list() const override;
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override;
  bool IdenticalTo(const std::shared_ptr<LazyConsistentTensorImpl>& rhs) const;

  int64_t numpy_size() const override { UNIMPLEMENTED(); }
  int64_t numpy_list_size() const override { UNIMPLEMENTED(); }
  std::shared_ptr<compatible_py::BlobObject> blob_object() const override { UNIMPLEMENTED(); }
};

class LazyMirroredTensorImpl : public MirroredTensorImpl {
 public:
  LazyMirroredTensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                   const std::shared_ptr<compatible_py::Distribute>& distribute);
  LazyMirroredTensorImpl(const LazyMirroredTensorImpl& impl) = default;
  ~LazyMirroredTensorImpl() = default;

  std::vector<std::shared_ptr<LazyConsistentTensorImpl>> sub_consistent_blob_list();
  virtual std::string get_mirror_shape_log_warning() const;
  std::shared_ptr<Shape> shape() const override;
  DataType dtype() const override;
  int64_t batch_axis() const override;
  int64_t split_axis() const override;
  bool is_dynamic() const override;
  bool is_tensor_list() const override;
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override;

  int64_t numpy_size() const override { UNIMPLEMENTED(); }
  int64_t numpy_list_size() const override { UNIMPLEMENTED(); }
  std::shared_ptr<compatible_py::BlobObject> blob_object() const override { UNIMPLEMENTED(); }

 private:
  std::vector<std::shared_ptr<LazyConsistentTensorImpl>> sub_consistent_blob_list_;
};

class EagerConsistentTensorImpl : public ConsistentTensorImpl {
 public:
  EagerConsistentTensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                      const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                      const std::shared_ptr<compatible_py::BlobRegister>& blob_register,
                      const std::string& job_name, const std::shared_ptr<compatible_py::Distribute>& distribute);
  ~EagerConsistentTensorImpl();  

  std::shared_ptr<Shape> shape() const override;
  DataType dtype() const override;
  int64_t batch_axis() const override;
  int64_t split_axis() const override;
  bool is_dynamic() const override;
  bool is_tensor_list() const override;
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override;
  int64_t parallel_size() override;
  int64_t numpy_size() const override;
  int64_t numpy_list_size() const override;
  std::shared_ptr<compatible_py::BlobObject> blob_object() const override;
  bool IdenticalTo(const std::shared_ptr<EagerConsistentTensorImpl>& rhs) const;

 private:
  int64_t parallel_size_;
  std::string logical_blob_name_;
  std::shared_ptr<compatible_py::RegisteredBlobAccess> registered_blob_access_;
};

class EagerMirroredTensorImpl :  public MirroredTensorImpl {
 public:
  EagerMirroredTensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                    const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                    const std::shared_ptr<compatible_py::BlobRegister>& blob_register, const std::string& job_name,
                    const std::shared_ptr<compatible_py::Distribute>& distribute);
  ~EagerMirroredTensorImpl();

  std::shared_ptr<Shape> shape() const override;
  DataType dtype() const override;
  int64_t batch_axis() const override;
  int64_t split_axis() const override;
  bool is_dynamic() const override;
  bool is_tensor_list() const override;
  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override;
  int64_t parallel_size() override;
  int64_t numpy_size() const override;
  int64_t numpy_list_size() const override;
  std::shared_ptr<compatible_py::BlobObject> blob_object() const override;
  bool IdenticalTo(const std::shared_ptr<EagerMirroredTensorImpl>& rhs) const;

 private:
  int64_t parallel_size_;
  std::string logical_blob_name_;
  std::shared_ptr<compatible_py::RegisteredBlobAccess> registered_blob_access_;
};

}
}

