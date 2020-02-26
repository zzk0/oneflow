from google.protobuf import descriptor
from onnx import ModelProto, GraphProto, NodeProto, OperatorSetIdProto
from onnx import helper, onnx_pb, TensorProto
from oneflow.python.oneflow_export import oneflow_export
import oneflow.core.common.data_type_pb2 as data_type_pb2


def SetOutputRemoteBlob4IR(outputs):
    global cur_job_lbn_2_output_remote_blob
    cur_job_lbn_2_output_remote_blob = {}
    for key, output in outputs.items():
        lbn = output.lbi.op_name + '/' + output.lbi.blob_name
        cur_job_lbn_2_output_remote_blob[lbn] = output
    

def ClearOutputRemoteBlob4IR():
    global cur_job_lbn_2_output_remote_blob
    cur_job_lbn_2_output_remote_blob = {}


def AddCurrentJobOpConf4IR(op_conf):
    global cur_job_op_confs
    cur_job_op_confs.append(op_conf)


def ClearIRNodes():
    global cur_job_op_confs
    cur_job_op_confs = []


def _of_op_type_case2onnx_op_type(op_type_case):
    return op_type_case

_enroll_input_bns_in_op = {'axpy_conf': 'y'}

_enroll_input_bns_in_grad_op = [
    'y', 'normalized', 'mean', 'inv_variance',
]
_enroll_output_bns = [
    'y', 'normalized', 'mean', 'inv_variance',
    'loss', 'loss_instance_num', 'reduction_coefficient', 'out', 'out_', 'acc', 'dx',
    'transpose_in', 'transpose_out', 'accuracy', 'db', 'bias_diff', 'filter_diff', 'in_diff',
    'y_indices', 'y_values', 'num_unique', 'beta_diff', 'gamma_diff', 'normalized_diff',
    'alpha_grad', 'prediction_diff', 'idx', 'count'
    #'y', 'normalized', 'mean', 'inv_variance',
]
_enroll_input_bns = [
    'prediction', 'label', 'weight', 'in', 'bias', 'moving_mean', 'moving_variance', 'gamma',
    'beta', 'tick', 'momentum', 'one', 'indices', 'mask', 'dy', 'like', 'm', 'v', 'beta1_t',
    'beta2_t', 'in_', 'a', 'b', 'model_diff', 'model', 'learning_rate', 'train_step', 'alpha',
    'path', 'in_0', 'in_1', 'ref', 'value', 'x', 'filter', 'x_like', 'out_diff',
    'model_diff_indices', 'model_diff_values', 'x_indices', 'x_values', 
    'start', 'end', 'loss_diff', 'transpose_x', 'transpose_y', 'segment_ids', 'data',
    #'y', 'normalized', 'mean', 'inv_variance',
]

def _is_in(op_type_case, field_name):
    if op_type_case in _enroll_input_bns_in_op:
        if _enroll_input_bns_in_op[op_type_case] == field_name:
            return True
    elif field_name in _enroll_input_bns_in_grad_op:
        return 'grad' in op_type_case
    elif field_name in _enroll_input_bns:
        return True
    else:
        return False
        

def _is_out(op_type_case, field_name):
    if op_type_case in _enroll_input_bns_in_op:
        if _enroll_input_bns_in_op[op_type_case] == field_name:
            return False
    elif field_name in _enroll_input_bns_in_grad_op:
        return 'grad' not in op_type_case
    elif field_name in _enroll_output_bns:
        return True 
    else:
        return False

def PrepareOnnxArgs(op_conf):
    op_type_case = op_conf.WhichOneof('op_type')
    print(op_type_case)
    op_type_pb2 = getattr(op_conf, op_type_case)
    op_type = _of_op_type_case2onnx_op_type(op_type_case)

    fields = op_type_pb2.ListFields()
    inputs = []
    outputs = []
    attr = {}
    try:
        for field, value in fields:
            #TODO repeated io: 
            if field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
                print("{} is repeated".format(field.name), value)
            if _is_in(op_type_case, field.name):
                inputs.append(value)
            elif _is_out(op_type_case, field.name):
                outputs.append('{}/{}'.format(op_conf.name, value))
            else:
                attr[field.name] = 'TBD'#value #TODO of field to onnx attr
    except ValueError as e:
        raise SerializeToJsonError('Failed to serialize {0} field: {1}.'.format(field.name, e))
    return op_type, inputs, outputs, attr


OF_TO_ONNX_DTYPE = {
    data_type_pb2.kFloat:TensorProto.FLOAT,
    data_type_pb2.kInt32:TensorProto.INT32,
    #TODO support more dtypes 
}

def Prepare4OnnxGraph():
    inputs = []
    outputs = []
    nodes = []
    initializers = []
    global cur_job_op_confs
    global cur_job_lbn_2_output_remote_blob
    for op_conf in cur_job_op_confs:
        op_type_case = op_conf.WhichOneof('op_type')
        op_type_pb2 = getattr(op_conf, op_type_case)
        if op_type_case == 'input_conf':
            lbn = op_conf.name + '/' + op_type_pb2.out
            dtype = OF_TO_ONNX_DTYPE[op_type_pb2.blob_conf.data_type]
            shape = op_type_pb2.blob_conf.shape.dim
            inputs.append(helper.make_tensor_value_info(lbn, dtype, shape))
        elif op_type_case == 'return_conf':
            lbn = op_conf.name + '/' + op_type_pb2.out
            assert lbn in cur_job_lbn_2_output_remote_blob
            dtype = OF_TO_ONNX_DTYPE[cur_job_lbn_2_output_remote_blob[lbn].dtype]
            shape = cur_job_lbn_2_output_remote_blob[lbn].shape 
            outputs.append(helper.make_tensor_value_info(lbn, dtype, shape))
            print('outputs', outputs[-1])
        elif op_type_case == 'variable_conf':
            #TODO
            print('initializers', op_conf)
        else:
            #TODO
            print('nodes', op_conf)
    return nodes, inputs, outputs, initializers

@oneflow_export('export_onnx')
def SaveOnnxModelProto(path='model.onnx'):
    opset_id = OperatorSetIdProto()
    opset_id.domain = ''
    opset_id.version = 9
    global cur_job_lbn_2_output_remote_blob
    job_name = ''
    for key, v in cur_job_lbn_2_output_remote_blob.items():
        print(dir(v))
        job_name = v.job_name_
        break

    nodes, inputs, outputs, initializers = Prepare4OnnxGraph()
    graph_pb2 = helper.make_graph(nodes, job_name, inputs, outputs, initializer=initializers)
    model = helper.make_model(graph_pb2, ir_version=4, opset_imports=[opset_id],
                              producer_name='oneflow')
    return model


global cur_job_op_confs
cur_job_op_confs = []

global cur_job_lbn_2_output_remote_blob
cur_job_lbn_2_output_remote_blob = {} 