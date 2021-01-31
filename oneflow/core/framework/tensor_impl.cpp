#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/common/global.h"

namespace oneflow {

namespace one {

static int64_t INVALID_BATCH_AXIS = -22;
static int64_t INVALID_SPLIT_AXIS = -22;

namespace {

Maybe<JobBuildAndInferCtxMgr*> GlobalJobBuildAndInferCtxMgr() {
  if (EagerExecutionEnabled()) {
    return JUST(GlobalMaybe<EagerJobBuildAndInferCtxMgr>());
  } else {
    return JUST(GlobalMaybe<LazyJobBuildAndInferCtxMgr>());
  }
}

Maybe<JobBuildAndInferCtx*> GetJobBuildAndInferCtx(const std::string& job_name) {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  return mgr->FindJobBuildAndInferCtx(job_name);
}

}  // namespace


TensorImpl::TensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                   const std::shared_ptr<compatible_py::Distribute>& distribute)
    : lbi_(lbi), distribute_(distribute) {
  lbn_ = lbi->op_name() + "/" + lbi->blob_name();
}

std::shared_ptr<cfg::LogicalBlobId> TensorImpl::lbi() const { return lbi_; }
std::string TensorImpl::logical_blob_name() const { return lbn_; }
std::string TensorImpl::op_name() const { return lbi_->op_name(); }
std::string TensorImpl::blob_name() const { return lbi_->blob_name(); }

int64_t TensorImpl::batch_axis() const { UNIMPLEMENTED(); }
bool TensorImpl::has_batch_axis() const { return batch_axis() != INVALID_BATCH_AXIS; }
std::shared_ptr<compatible_py::Distribute> TensorImpl::distribute() const { return distribute_; }
std::string TensorImpl::unique_name() const { return lbn_ + *CHECK_JUST(Distribute2Str()); }

void TensorImpl::set_distribute(const std::shared_ptr<compatible_py::Distribute> distribute) {
  distribute_ = distribute;
}

Maybe<std::string> TensorImpl::Distribute2Str() const {
  if (std::dynamic_pointer_cast<compatible_py::AutoDistribute>(distribute_)) {
    return std::string("");
  } else if (std::dynamic_pointer_cast<compatible_py::BroadcastDistribute>(distribute_)) {
    return std::string(":B");
  } else if (std::dynamic_pointer_cast<compatible_py::SplitDistribute>(distribute_)) {
    return std::string(":S") + std::to_string(distribute_->axis());
  } else {
    OF_UNIMPLEMENTED();
  }
  return std::string("");
}

ConsistentTensorImpl::ConsistentTensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                               const std::string& job_name,
                               const std::shared_ptr<compatible_py::Distribute>& distribute)
    : TensorImpl(lbi, distribute), parallel_size_(0) {
  if (job_name.empty()) {
    auto* mgr = CHECK_JUST(GlobalJobBuildAndInferCtxMgr());
    job_name_ = *CHECK_JUST(mgr->GetCurrentJobName());
  } else {
    job_name_ = job_name;
  }
}

std::string ConsistentTensorImpl::job_name() const { return job_name_; }

int64_t ConsistentTensorImpl::parallel_size() {
  if (parallel_size_ == 0) {
    std::shared_ptr<cfg::ParallelConf> cfg_parallel_conf = parallel_conf();
    ParallelConf proto_parallel_conf;
    cfg_parallel_conf->ToProto(&proto_parallel_conf);
    ParallelDesc parallel_desc(proto_parallel_conf);
    parallel_size_ = parallel_desc.parallel_num();
  }
  return parallel_size_;
}

void ConsistentTensorImpl::set_job_name(std::string job_name) { job_name_ = job_name; }

MirroredTensorImpl::MirroredTensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                           const std::string& job_name,
                           const std::shared_ptr<compatible_py::Distribute>& distribute)
    : TensorImpl(lbi, distribute), parallel_size_(0) {
  if (job_name.empty()) {
    auto* mgr = CHECK_JUST(GlobalJobBuildAndInferCtxMgr());
    job_name_ = *CHECK_JUST(mgr->GetCurrentJobName());
  } else {
    job_name_ = job_name;
  }
}

std::string MirroredTensorImpl::job_name() const { return job_name_; }

int64_t MirroredTensorImpl::parallel_size() {
  if (parallel_size_ == 0) {
    std::shared_ptr<cfg::ParallelConf> cfg_parallel_conf = parallel_conf();
    ParallelConf proto_parallel_conf;
    cfg_parallel_conf->ToProto(&proto_parallel_conf);
    ParallelDesc parallel_desc(proto_parallel_conf);
    parallel_size_ = parallel_desc.parallel_num();
  }
  return parallel_size_;
}

void MirroredTensorImpl::set_job_name(std::string job_name) { job_name_ = job_name; }

LazyConsistentTensorImpl::LazyConsistentTensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                                       const std::string& job_name,
                                       const std::shared_ptr<compatible_py::Distribute>& distribute)
    : ConsistentTensorImpl(lbi, job_name, distribute) {}

std::string LazyConsistentTensorImpl::get_lazy_shape_log_warning() const { return std::string(""); }

std::shared_ptr<Shape> LazyConsistentTensorImpl::shape() const {
  const std::string& log_warning = get_lazy_shape_log_warning();
  if (!log_warning.empty()) { LOG(ERROR) << log_warning; }
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->GetStaticShape(logical_blob_name()));
}

DataType LazyConsistentTensorImpl::dtype() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->GetDataType(logical_blob_name()));
}

int64_t LazyConsistentTensorImpl::batch_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->GetBatchAxis(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_BATCH_AXIS;
}

int64_t LazyConsistentTensorImpl::split_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->GetSplitAxisFromProducerView(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_SPLIT_AXIS;
}

bool LazyConsistentTensorImpl::is_dynamic() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->IsDynamic(logical_blob_name()));
}

bool LazyConsistentTensorImpl::is_tensor_list() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->IsTensorList(logical_blob_name()));
}

std::shared_ptr<cfg::ParallelConf> LazyConsistentTensorImpl::parallel_conf() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->GetParallelDescFromProducerView(logical_blob_name()))->cfg_parallel_conf();
}

bool LazyConsistentTensorImpl::IdenticalTo(const std::shared_ptr<LazyConsistentTensorImpl>& rhs) const {
  return true && unique_name() == rhs->unique_name() && *shape() == *rhs->shape()
         && batch_axis() == rhs->batch_axis() && split_axis() == rhs->split_axis()
         && is_dynamic() == rhs->is_dynamic() && is_tensor_list() == rhs->is_tensor_list();
}

LazyMirroredTensorImpl::LazyMirroredTensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                   const std::shared_ptr<compatible_py::Distribute>& distribute) : MirroredTensorImpl(lbi, job_name, distribute){
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(this->job_name()));
  int lbi_num = CHECK_JUST(ctx->MirroredBlobGetNumSubLbi(this->logical_blob_name()));
  for (int i = 0; i < lbi_num; ++i) {
    std::shared_ptr<cfg::LogicalBlobId> sub_lbi = std::make_shared<cfg::LogicalBlobId>(
        *CHECK_JUST(ctx->MirroredBlobGetSubLbi(this->logical_blob_name(), i)));
    sub_consistent_blob_list_.emplace_back(
        std::make_shared<LazyConsistentTensorImpl>(sub_lbi, "", compatible_py::GlobalAutoDistribute()));
  }
}

std::vector<std::shared_ptr<LazyConsistentTensorImpl>> LazyMirroredTensorImpl::sub_consistent_blob_list() {
  return sub_consistent_blob_list_;
}

std::string LazyMirroredTensorImpl::get_mirror_shape_log_warning() const { return std::string(""); }

std::shared_ptr<Shape> LazyMirroredTensorImpl::shape() const {
  const std::string& log_warning = get_mirror_shape_log_warning();
  if (!log_warning.empty()) { LOG(ERROR) << log_warning; }
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto shape = CHECK_JUST(ctx->MirroredBlobGetStaticShape(logical_blob_name()));
  return shape;
}

DataType LazyMirroredTensorImpl::dtype() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobGetDataType(logical_blob_name()));
}

int64_t LazyMirroredTensorImpl::batch_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->MirroredBlobGetBatchAxis(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_BATCH_AXIS;
}

int64_t LazyMirroredTensorImpl::split_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->MirroredBlobGetSplitAxisFromProducerView(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_SPLIT_AXIS;
}

bool LazyMirroredTensorImpl::is_dynamic() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobIsDynamic(logical_blob_name()));
}

bool LazyMirroredTensorImpl::is_tensor_list() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobIsTensorList(logical_blob_name()));
}

std::shared_ptr<cfg::ParallelConf> LazyMirroredTensorImpl::parallel_conf() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobGetParallelDescFromProducerView(logical_blob_name()))
      ->cfg_parallel_conf();
}

EagerConsistentTensorImpl::EagerConsistentTensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                                         const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                                         const std::shared_ptr<compatible_py::BlobRegister>& blob_register,
                                         const std::string& job_name,
                                         const std::shared_ptr<compatible_py::Distribute>& distribute)
    : ConsistentTensorImpl(lbi, job_name, distribute), parallel_size_(0) {
  std::string logical_blob_name = lbi->op_name() + "/" + lbi->blob_name();
  std::shared_ptr<compatible_py::RegisteredBlobAccess> access =
      blob_register->OpenRegisteredBlobAccess(logical_blob_name, blob_object);
  registered_blob_access_ = access;
}

EagerConsistentTensorImpl::~EagerConsistentTensorImpl() {
  registered_blob_access_->blob_register()->CloseRegisteredBlobAccess(logical_blob_name_);
}

int64_t EagerConsistentTensorImpl::numpy_size() const {
  return blob_object()->parallel_desc_symbol()->parallel_num();
}

int64_t EagerConsistentTensorImpl::numpy_list_size() const {
  return blob_object()->parallel_desc_symbol()->parallel_num();
}

std::shared_ptr<Shape> EagerConsistentTensorImpl::shape() const {
  return blob_object()->op_arg_blob_attr()->shape();
}

DataType EagerConsistentTensorImpl::dtype() const {
  return static_cast<DataType>(blob_object()->op_arg_blob_attr()->get_dtype());
}

int64_t EagerConsistentTensorImpl::batch_axis() const {
  auto opt_batch_axis = blob_object()->op_arg_blob_attr()->batch_axis();
  if (opt_batch_axis->has_value()) {
    return opt_batch_axis->value();
  } else {
    return INVALID_BATCH_AXIS;
  }
}

int64_t EagerConsistentTensorImpl::split_axis() const {
  auto sbp_parallel = blob_object()->op_arg_parallel_attr()->sbp_parallel();
  if (sbp_parallel->has_split_parallel()) {
    return sbp_parallel->split_parallel().axis();
  } else if (sbp_parallel->has_broadcast_parallel()) {
    return INVALID_SPLIT_AXIS;
  } else if (sbp_parallel->has_partial_sum_parallel()) {
    return INVALID_SPLIT_AXIS;
  } else {
    UNIMPLEMENTED();
  }
}

bool EagerConsistentTensorImpl::is_dynamic() const { return blob_object()->op_arg_blob_attr()->is_dynamic(); }

bool EagerConsistentTensorImpl::is_tensor_list() const {
  return blob_object()->op_arg_blob_attr()->is_tensor_list();
}

std::shared_ptr<cfg::ParallelConf> EagerConsistentTensorImpl::parallel_conf() const {
  return blob_object()->parallel_desc_symbol()->cfg_parallel_conf();
}

int64_t EagerConsistentTensorImpl::parallel_size() {
  if (parallel_size_ == 0) {
    std::shared_ptr<cfg::ParallelConf> cfg_parallel_conf = parallel_conf();
    ParallelConf proto_parallel_conf;
    cfg_parallel_conf->ToProto(&proto_parallel_conf);
    ParallelDesc parallel_desc(proto_parallel_conf);
    parallel_size_ = parallel_desc.parallel_num();
  }
  return parallel_size_;
}

std::shared_ptr<compatible_py::BlobObject> EagerConsistentTensorImpl::blob_object() const {
  return registered_blob_access_->blob_object();
}

bool EagerConsistentTensorImpl::IdenticalTo(const std::shared_ptr<EagerConsistentTensorImpl>& rhs) const {
  return (blob_object()->op_arg_blob_attr() == rhs->blob_object()->op_arg_blob_attr())
         && (blob_object()->op_arg_parallel_attr() == rhs->blob_object()->op_arg_parallel_attr());
}

EagerMirroredTensorImpl::EagerMirroredTensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                                     const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                                     const std::shared_ptr<compatible_py::BlobRegister>& blob_register,
                                     const std::string& job_name,
                                     const std::shared_ptr<compatible_py::Distribute>& distribute)
    : MirroredTensorImpl(lbi, job_name, distribute), parallel_size_(0) {
  std::string logical_blob_name = lbi->op_name() + "/" + lbi->blob_name();
  std::shared_ptr<compatible_py::RegisteredBlobAccess> access =
      blob_register->OpenRegisteredBlobAccess(logical_blob_name, blob_object);
  registered_blob_access_ = access;
}

EagerMirroredTensorImpl::~EagerMirroredTensorImpl() {
  registered_blob_access_->blob_register()->CloseRegisteredBlobAccess(logical_blob_name_);
}

int64_t EagerMirroredTensorImpl::numpy_size() const {
  return blob_object()->parallel_desc_symbol()->parallel_num();
}

int64_t EagerMirroredTensorImpl::numpy_list_size() const {
  return blob_object()->parallel_desc_symbol()->parallel_num();
}

std::shared_ptr<Shape> EagerMirroredTensorImpl::shape() const {
  return blob_object()->op_arg_blob_attr()->shape();
}

DataType EagerMirroredTensorImpl::dtype() const {
  return static_cast<DataType>(blob_object()->op_arg_blob_attr()->get_dtype());
}

int64_t EagerMirroredTensorImpl::batch_axis() const {
  auto opt_batch_axis = blob_object()->op_arg_blob_attr()->batch_axis();
  if (opt_batch_axis->has_value()) {
    return opt_batch_axis->value();
  } else {
    return INVALID_BATCH_AXIS;
  }
}

int64_t EagerMirroredTensorImpl::split_axis() const {
  auto sbp_parallel = blob_object()->op_arg_parallel_attr()->sbp_parallel();
  if (sbp_parallel->has_split_parallel()) {
    return sbp_parallel->split_parallel().axis();
  } else if (sbp_parallel->has_broadcast_parallel()) {
    return INVALID_SPLIT_AXIS;
  } else if (sbp_parallel->has_partial_sum_parallel()) {
    return INVALID_SPLIT_AXIS;
  } else {
    UNIMPLEMENTED();
  }
}

bool EagerMirroredTensorImpl::is_dynamic() const { return blob_object()->op_arg_blob_attr()->is_dynamic(); }

bool EagerMirroredTensorImpl::is_tensor_list() const {
  return blob_object()->op_arg_blob_attr()->is_tensor_list();
}

std::shared_ptr<cfg::ParallelConf> EagerMirroredTensorImpl::parallel_conf() const {
  return blob_object()->parallel_desc_symbol()->cfg_parallel_conf();
}

int64_t EagerMirroredTensorImpl::parallel_size() {
  if (parallel_size_ == 0) {
    std::shared_ptr<cfg::ParallelConf> cfg_parallel_conf = parallel_conf();
    ParallelConf proto_parallel_conf;
    cfg_parallel_conf->ToProto(&proto_parallel_conf);
    ParallelDesc parallel_desc(proto_parallel_conf);
    parallel_size_ = parallel_desc.parallel_num();
  }
  return parallel_size_;
}

std::shared_ptr<compatible_py::BlobObject> EagerMirroredTensorImpl::blob_object() const {
  return registered_blob_access_->blob_object();
}

bool EagerMirroredTensorImpl::IdenticalTo(const std::shared_ptr<EagerMirroredTensorImpl>& rhs) const {
  return (blob_object()->op_arg_blob_attr() == rhs->blob_object()->op_arg_blob_attr())
         && (blob_object()->op_arg_parallel_attr() == rhs->blob_object()->op_arg_parallel_attr());
}
}

}

