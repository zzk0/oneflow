#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/actor/msg_event.pb.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

OF_DEFINE_ENUM_TO_OSTREAM_FUNC(ActorCmd);
OF_DEFINE_ENUM_TO_OSTREAM_FUNC(ActorMsgType);

void LogToConsumerMsg(const ActorMsg& msg) {
  if (Global<RuntimeCtx>::Get()->is_experiment_phase()) {
    if (msg.msg_type() == ActorMsgType::kRegstMsg) {
      Regst* regst = msg.regst();
      // get nanoseconds, e.g. 1505840189520477525 = 1505840189.520477525 sec
      int64_t start =
          std::chrono::high_resolution_clock::now().time_since_epoch().count();
      MsgEvent* msg_event = nullptr;
      msg_event = new MsgEvent;
      msg_event->set_time(start);
      msg_event->set_src_actor_id(msg.src_actor_id());
      msg_event->set_dst_actor_id(msg.dst_actor_id());
      msg_event->set_producer_actor_id(regst->producer_actor_id());
      msg_event->set_act_id(msg.act_id());
      msg_event->set_model_version_id(regst->model_version_id());
      msg_event->set_piece_id(regst->piece_id());
      // msg_event->set_model_version_id(regst->model_version_id());
      msg_event->set_regst_desc_id(regst->regst_desc_id());
      msg_event->set_info("to_consumer");
      Global<CtrlClient>::Get()->PushMsgEvent(*msg_event);
      delete msg_event;
    }
  }
}

ActorMsg ActorMsg::BuildRegstMsgToConsumer(int64_t producer, int64_t consumer,
                                           Regst* regst_raw_ptr) {
  ActorMsg msg;
  msg.src_actor_id_ = producer;
  msg.dst_actor_id_ = consumer;
  msg.msg_type_ = ActorMsgType::kRegstMsg;
  msg.regst_wrapper_.regst = regst_raw_ptr;
  if (Global<IDMgr>::Get()->MachineId4ActorId(consumer)
      == Global<MachineCtx>::Get()->this_machine_id()) {
    msg.regst_wrapper_.comm_net_token = nullptr;
  } else {
    msg.regst_wrapper_.comm_net_token =
        regst_raw_ptr->packed_blob()->comm_net_token();
  }
  msg.regst_wrapper_.regst_status = regst_raw_ptr->status();
  msg.regst_wrapper_.regst_status.regst_desc_id =
      regst_raw_ptr->regst_desc_id();
  LogToConsumerMsg(msg);
  return msg;
}

ActorMsg ActorMsg::BuildRegstMsgToProducer(int64_t consumer, int64_t producer,
                                           Regst* regst_raw_ptr) {
  ActorMsg msg;
  msg.src_actor_id_ = consumer;
  msg.dst_actor_id_ = producer;
  msg.msg_type_ = ActorMsgType::kRegstMsg;
  msg.regst_wrapper_.regst = regst_raw_ptr;
  msg.regst_wrapper_.comm_net_token = nullptr;
  return msg;
}

ActorMsg ActorMsg::BuildEordMsg(int64_t consumer, int64_t regst_desc_id) {
  ActorMsg msg;
  msg.src_actor_id_ = -1;
  msg.dst_actor_id_ = consumer;
  msg.msg_type_ = ActorMsgType::kEordMsg;
  msg.eord_regst_desc_id_ = regst_desc_id;
  return msg;
}

ActorMsg ActorMsg::BuildCommandMsg(int64_t dst_actor_id, ActorCmd cmd) {
  ActorMsg msg;
  msg.src_actor_id_ = -1;
  msg.dst_actor_id_ = dst_actor_id;
  msg.msg_type_ = ActorMsgType::kCmdMsg;
  msg.actor_cmd_ = cmd;
  return msg;
}

int64_t ActorMsg::SrcMachineId() const {
  return Global<IDMgr>::Get()->MachineId4ActorId(src_actor_id_);
}

ActorCmd ActorMsg::actor_cmd() const {
  CHECK_EQ(msg_type_, ActorMsgType::kCmdMsg);
  return actor_cmd_;
}

Regst* ActorMsg::regst() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.regst;
}

int64_t ActorMsg::piece_id() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.regst_status.piece_id;
}

int64_t ActorMsg::act_id() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.regst_status.act_id;
}

int64_t ActorMsg::model_version_id() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.regst_status.model_version_id;
}

int64_t ActorMsg::regst_desc_id() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.regst_status.regst_desc_id;
}

const void* ActorMsg::comm_net_token() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.comm_net_token;
}

int64_t ActorMsg::eord_regst_desc_id() const {
  CHECK_EQ(msg_type_, ActorMsgType::kEordMsg);
  return eord_regst_desc_id_;
}

}  // namespace oneflow
