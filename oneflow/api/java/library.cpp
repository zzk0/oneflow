#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <future>

#include "jni.h"
#include "jni_md.h"
#include "oneflow/api/java/library.h"
#include "oneflow/api/python/framework/framework.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/api/python/session/session.h"
#include "oneflow/api/python/env/env.h"
#include "oneflow/api/python/session/session_api.h"
#include "oneflow/core/common/cfg.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/api/java/util/jni_util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/vm/init_symbol_instruction_type.h"
#include "oneflow/core/framework/symbol_id_cache.h"
#include "oneflow/core/vm/stream_type.h"

JNIEXPORT 
jint JNICALL Java_org_oneflow_InferenceSession_getEndian(JNIEnv* env, jobject obj) {
  return Endian();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_initDefaultSession(JNIEnv* env, jobject obj) {
  int64_t session_id = oneflow::NewSessionId();
  oneflow::RegsiterSession(session_id);
}

JNIEXPORT
jboolean JNICALL Java_org_oneflow_InferenceSession_isEnvInited(JNIEnv* env, jobject obj) {
  return oneflow::IsEnvInited().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_initEnv(JNIEnv* env, jobject obj, jstring env_proto_str) {
  oneflow::InitEnv(ConvertToString(env, env_proto_str), false).GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_initScopeStack(JNIEnv* env, jobject obj, jstring jstr) {
  std::shared_ptr<oneflow::cfg::JobConfigProto> job_conf = std::make_shared<oneflow::cfg::JobConfigProto>();
  job_conf->mutable_predict_conf();
  job_conf->set_job_name("");

  std::shared_ptr<oneflow::Scope> scope;
  auto BuildInitialScope = [&scope, &job_conf](oneflow::InstructionsBuilder* builder) mutable -> oneflow::Maybe<void> {
    int session_id = oneflow::GetDefaultSessionId().GetOrThrow();
    const std::vector<std::string> machine_device_ids({"0:0"});
    std::shared_ptr<oneflow::Scope> initialScope = builder->BuildInitialScope(session_id, job_conf, "cpu", machine_device_ids, nullptr, false).GetPtrOrThrow();
    scope = initialScope;
    return oneflow::Maybe<void>::Ok();
  };
  oneflow::LogicalRun(BuildInitialScope);
  oneflow::InitThreadLocalScopeStack(scope);  // fixme: bug? Is LogicalRun asynchronous or synchronous?
}

JNIEXPORT
jboolean JNICALL Java_org_oneflow_InferenceSession_isSessionInited(JNIEnv* env, jobject obj) {
  return oneflow::IsSessionInited().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_initSession(JNIEnv* env, jobject obj) {
  // default configuration, Todo: configuration
  // reference: oneflow/python/framework/session_util.py
  std::shared_ptr<oneflow::ConfigProto> config_proto = std::make_shared<oneflow::ConfigProto>();
  config_proto->mutable_resource()->set_machine_num(oneflow::GetNodeSize().GetOrThrow());
  config_proto->mutable_resource()->set_gpu_device_num(1);
  config_proto->set_session_id(oneflow::GetDefaultSessionId().GetOrThrow());
  config_proto->mutable_resource()->set_enable_legacy_model_io(true);

  oneflow::InitLazyGlobalSession(config_proto->DebugString());
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_openJobBuildAndInferCtx(JNIEnv* env, jobject obj, jstring job_name) {
  oneflow::SetIsMultiClient(false);
  oneflow::JobBuildAndInferCtx_Open(ConvertToString(env, job_name)).GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_setJobConfForCurJobBuildAndInferCtx(JNIEnv* env, jobject obj, jstring job_conf_proto) {
  oneflow::JobConfigProto job_conf;
  std::string job_conf_txt = ConvertToString(env, job_conf_proto);
  oneflow::TxtString2PbMessage(job_conf_txt, &job_conf);

  oneflow::cfg::JobConfigProto job_conf_cfg;
  job_conf_cfg.InitFromProto(job_conf);
  oneflow::CurJobBuildAndInferCtx_SetJobConf(job_conf_cfg).GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_setScopeForCurJob(JNIEnv* env, jobject obj, jstring jstr) {
  oneflow::JobConfigProto job_conf;
  std::string job_conf_txt = ConvertToString(env, jstr);
  oneflow::TxtString2PbMessage(job_conf_txt, &job_conf);

  std::shared_ptr<oneflow::cfg::JobConfigProto> job_conf_cfg = std::make_shared<oneflow::cfg::JobConfigProto>();
  job_conf_cfg->InitFromProto(job_conf);

  std::shared_ptr<oneflow::Scope> scope;
  auto BuildInitialScope = [&scope, &job_conf_cfg](oneflow::InstructionsBuilder* builder) mutable -> oneflow::Maybe<void> {
    int session_id = oneflow::GetDefaultSessionId().GetOrThrow();
    const std::vector<std::string> machine_device_ids({"0:0"});
    std::shared_ptr<oneflow::Scope> initialScope = builder->BuildInitialScope(session_id, job_conf_cfg, "gpu", machine_device_ids, nullptr, false).GetPtrOrThrow();
    scope = initialScope;
    return oneflow::Maybe<void>::Ok();
  };
  oneflow::LogicalRun(BuildInitialScope);
  oneflow::ThreadLocalScopeStackPush(scope).GetOrThrow();  // fixme: bug?
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_curJobAddOp(JNIEnv* env, jobject obj, jstring op_conf_proto) {
  std::string op_conf_proto_str = ConvertToString(env, op_conf_proto);
  op_conf_proto_str = Subreplace(op_conf_proto_str, "user_input", "input");
  op_conf_proto_str = Subreplace(op_conf_proto_str, "user_output", "output");

  oneflow::OperatorConf op_conf;
  oneflow::TxtString2PbMessage(op_conf_proto_str, &op_conf);

  auto scope = oneflow::GetCurrentScope().GetPtrOrThrow();
  op_conf.set_scope_symbol_id(scope->symbol_id().GetOrThrow());
  op_conf.set_device_tag(scope->device_parallel_desc_symbol()->device_tag());
  oneflow::CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf.DebugString());
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_completeCurJobBuildAndInferCtx(JNIEnv* env, jobject obj) {
  oneflow::CurJobBuildAndInferCtx_Complete().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_rebuildCurJobBuildAndInferCtx(JNIEnv* env, jobject obj) {
  oneflow::CurJobBuildAndInferCtx_Rebuild().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_unsetScopeForCurJob(JNIEnv* env, jobject obj) {
  oneflow::ThreadLocalScopeStackPop().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_closeJobBuildAndInferCtx(JNIEnv* env, jobject obj) {
  oneflow::JobBuildAndInferCtx_Close().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_startLazyGlobalSession(JNIEnv* env, jobject obj) {
  oneflow::StartLazyGlobalSession().GetOrThrow();
}

namespace oneflow {

class JavaForeignJobInstance : public JobInstance {
 public:
  JavaForeignJobInstance(std::string job_name,  std::string sole_input_op_name_in_user_job,
                         std::string sole_output_op_name_in_user_job, std::function<void(uint64_t)> push_cb,
                         std::function<void(uint64_t)> pull_cb, std::function<void()> finish) : 
                           job_name_(job_name), 
                           sole_input_op_name_in_user_job_(sole_input_op_name_in_user_job),
                           sole_output_op_name_in_user_job_(sole_output_op_name_in_user_job),
                           push_cb_(push_cb),
                           pull_cb_(pull_cb),
                           finish_(finish) {
  }
  ~JavaForeignJobInstance() {}
  std::string job_name() const { return job_name_; }
  std::string sole_input_op_name_in_user_job() const { return sole_input_op_name_in_user_job_; }
  std::string sole_output_op_name_in_user_job() const { return sole_output_op_name_in_user_job_; }
  void PushBlob(uint64_t ofblob_ptr) const {
    if (push_cb_ != nullptr) push_cb_(ofblob_ptr);
  }
  void PullBlob(uint64_t ofblob_ptr) const {
    if (pull_cb_ != nullptr) pull_cb_(ofblob_ptr);
  }
  void Finish() const {
    if (finish_ != nullptr) finish_();
  }

 private:
  std::string job_name_;
  std::string sole_input_op_name_in_user_job_;
  std::string sole_output_op_name_in_user_job_;
  std::function<void(uint64_t)> push_cb_;
  std::function<void(uint64_t)> pull_cb_;
  std::function<void()> finish_;
};

}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_loadCheckpoint(JNIEnv* env, jobject obj, jstring load_job, jobject path) {
  std::string load_job_name = ConvertToString(env, load_job);
  
  int _path_length = (*env).GetDirectBufferCapacity(path);
  int64_t *_shape = new int64_t[1]{ _path_length };  // Todo: is there a better way to allocate memory?
  void *_path = (*env).GetDirectBufferAddress(path);

  auto copy_model_load_path = [_shape, _path, _path_length](uint64_t of_blob_ptr) -> void {
    using namespace oneflow;
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    of_blob->CopyShapeFrom(_shape, 1);
    of_blob->AutoMemCopyFrom((signed char*) _path, _path_length);

    delete []_shape;
  };
  const std::shared_ptr<oneflow::JobInstance> job_inst(
    new oneflow::JavaForeignJobInstance(load_job_name, "", "", copy_model_load_path, nullptr, nullptr)
  );
  oneflow::LaunchJob(job_inst);
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_runSinglePushJob(JNIEnv* env, jobject obj, jobject data, jobject shape, jint dtype_code, jstring job_name, jstring op_name) {
  // get address and length
  void *data_arr = (*env).GetDirectBufferAddress(data);

  // copy shape
  long *shape_arr = (long*) (*env).GetDirectBufferAddress(shape);
  long shape_arr_length = (*env).GetDirectBufferCapacity(shape);

  // number of elements
  long element_number = 1;
  for (int i = 0; i < shape_arr_length; i++) {
    element_number = element_number * shape_arr[i];
  }

  // job_name & op_name
  std::string _job_name = ConvertToString(env, job_name);
  std::string _op_name = ConvertToString(env, op_name);
  
  auto job_instance_fun = [=](uint64_t of_blob_ptr) -> void {
    using namespace oneflow;
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    of_blob->CopyShapeFrom(shape_arr, shape_arr_length);
    if (dtype_code == kFloat) {
      of_blob->AutoMemCopyFrom((float*) data_arr, element_number);
    }
    if (dtype_code == kInt32) {
      of_blob->AutoMemCopyFrom((int*) data_arr, element_number);
    }
  };
  const std::shared_ptr<oneflow::JobInstance> job_instance(
    new oneflow::JavaForeignJobInstance(_job_name, _op_name, "", job_instance_fun, nullptr, nullptr)
  );
  oneflow::LaunchJob(job_instance);
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_runInferenceJob(JNIEnv* env, jobject obj, jstring jstr) {
  std::string inference_job_name = ConvertToString(env, jstr);
  const std::shared_ptr<oneflow::JobInstance> job_inst(
    new oneflow::JavaForeignJobInstance(inference_job_name, "", "", nullptr, nullptr, nullptr)
  );
  oneflow::LaunchJob(job_inst);
}

JNIEXPORT
jobject JNICALL Java_org_oneflow_InferenceSession_runPullJob(JNIEnv* env, jobject obj, jstring job_name, jstring op_name) {
  std::promise<unsigned char*> prom;
  std::future<unsigned char*> fut = prom.get_future();
  std::promise<uint64_t> len_prom;
  std::future<uint64_t> len_fut = len_prom.get_future();
  std::promise<long*> shape_prom;
  std::future<long*> shape_fut = shape_prom.get_future();
  std::promise<size_t> axes_prom;
  std::future<size_t> axes_fut = axes_prom.get_future();
  std::promise<int> dtype_prom;
  std::future<int> dtype_fut = dtype_prom.get_future();

  auto return_17 = [&](uint64_t of_blob_ptr) -> void {
    using namespace oneflow;
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
    
    size_t axes = of_blob->NumAxes();
    int64_t* shape = new int64_t[axes];
    of_blob->CopyShapeTo(shape, axes);

    int64_t element_number = 1;
    for (int i = 0; i < axes; i++) {
      element_number = element_number * shape[i];
    }
    if (of_blob->data_type() == kFloat) {
      element_number = element_number * 4;
    }
    if (of_blob->data_type() == kInt32) {
      element_number = element_number * 4;
    }
    unsigned char* data = new unsigned char[element_number];

    if (of_blob->data_type() == kFloat) {
      of_blob->AutoMemCopyTo((float*) data, element_number / 4);
    }
    if (of_blob->data_type() == kInt32) {
      of_blob->AutoMemCopyTo((int*) data, element_number / 4);
    }

    prom.set_value(data);
    len_prom.set_value(element_number);
    shape_prom.set_value(shape);
    dtype_prom.set_value(of_blob->data_type());
    axes_prom.set_value(axes);
  };

  std::string job_name_ = ConvertToString(env, job_name);
  std::string op_name_ = ConvertToString(env, op_name);

  const std::shared_ptr<oneflow::JobInstance> job_inst_return_17(
    new oneflow::JavaForeignJobInstance(job_name_, "", op_name_, nullptr, return_17, nullptr)
  );
  oneflow::LaunchJob(job_inst_return_17);

  unsigned char* data = fut.get();
  uint64_t len = len_fut.get();
  long* shape = shape_fut.get();
  int dtype = dtype_fut.get();
  size_t axes = axes_fut.get();

  jbyteArray array = (*env).NewByteArray(len);
  (*env).SetByteArrayRegion(array, 0, len, reinterpret_cast<jbyte*>(data));
  jlongArray shapeArray = (*env).NewLongArray(axes);
  (*env).SetLongArrayRegion(shapeArray, 0, axes, shape);

  // call nativeNewTensor
  jclass tensorClass = (*env).FindClass("org/oneflow/Tensor");
  jmethodID mid = (*env).GetStaticMethodID(tensorClass, "nativeNewTensor", "([B[JI)Lorg/oneflow/Tensor;");
  jobject tensor = (*env).CallStaticObjectMethod(tensorClass, mid, array, shapeArray, dtype);

  delete []data;
  delete []shape;
  return tensor;
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_stopLazyGlobalSession(JNIEnv* env, jobject obj) {
  oneflow::StopLazyGlobalSession().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_destroyLazyGlobalSession(JNIEnv* env, jobject obj) {
  oneflow::DestroyLazyGlobalSession().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_destroyEnv(JNIEnv* env, jobject obj) {
  oneflow::DestroyEnv().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_setShuttingDown(JNIEnv* env, jobject obj) {
  oneflow::SetShuttingDown();
}

JNIEXPORT
jstring JNICALL Java_org_oneflow_InferenceSession_getInterUserJobInfo(JNIEnv* env, jobject obj) {
  std::string inter_user_job_info = oneflow::GetSerializedInterUserJobInfo().GetOrThrow();
  return ConvertToJString(env, inter_user_job_info);
}
