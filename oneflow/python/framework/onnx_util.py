import os
import numpy as np
from google.protobuf import descriptor, text_format
from onnx import ModelProto, GraphProto, NodeProto, OperatorSetIdProto
from onnx import checker, helper, onnx_pb, TensorProto
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


def save_protobuf(path, message, as_text=False):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if as_text:
        with open(path, "w") as f:
            f.write(text_format.MessageToString(message))
    else:
        with open(path, "wb") as f:
            f.write(message.SerializeToString())


OF_TO_ONNX_DTYPE = {
    data_type_pb2.kFloat:TensorProto.FLOAT,
    data_type_pb2.kInt32:TensorProto.INT32,
    #TODO support more dtypes 
}

OF_TO_NUMPY_DTYPE = {
    data_type_pb2.kFloat:np.float32,
    data_type_pb2.kInt32:np.int32,
    #TODO support more dtypes 
}

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


OF_OP_TYPE2ONNX_OP_TYPE = {
    'reshape_conf': 'Flatten', #onnx reshape has two inputs: data and shape, shape is not attr. 
    'matmul_conf': 'Gemm', #onnx matmul does not supoort transpose a/b
    'conv_2d_conf': 'Conv', #onnx does not support bias_add
    'max_pooling_2d_conf': 'MaxPool',
    'average_pooling_2d_conf': 'AveragePool',
}

def _get_onnx_op_type(op_type_case):
    if op_type_case in OF_OP_TYPE2ONNX_OP_TYPE:
        return OF_OP_TYPE2ONNX_OP_TYPE[op_type_case]
    else:
        return op_type_case[:-5].title().replace('_', '')


def _of_field2onnx_attr(op_type, field, value):
    #if field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
    #    print("{} is repeated".format(field.name), value)
    takeit = True
    if op_type == 'Conv' or 'Pool' in op_type:
        if field.name == 'padding':
            #onnx support NOTSET, SAME_UPPER, SAME_LOWER or VALID
            return True, 'auto_pad', 'SAME_UPPER' if value.lower() != 'valid' else 'VALID'
        elif field.name == 'kernel_size' or field.name == 'pool_size':
            return True, 'kernel_shape', value
        elif field.name == 'dilation_rate':
            return True, 'dilations', value
        elif field.name == 'strides':
            return True, 'strides', value
        else:
            takeit = False # ignore other fields
    elif op_type == 'Flatten': #use flatten replace reshape
        return True, 'axis', 1
    elif op_type == 'Gemm': #oneflow matmul
        if field.name == 'transpose_a':
            return True, 'transA', 1 if value else 0 
        elif field.name == 'transpose_b':
            return True, 'transB', 1 if value else 0 
    return takeit, field.name, value 


def _op_conf2onnx_node(op_conf):
    # there are 2 options:
    # 1. convert of op_type_case to onnx one bye one
    # 2. convert of field to onnx attr one bye one, this may be better may be dirty somewhere
    op_type_case = op_conf.WhichOneof('op_type')
    op_type_pb2 = getattr(op_conf, op_type_case)
    onnx_op_type = _get_onnx_op_type(op_type_case)

    fields = op_type_pb2.ListFields()
    input_names = []
    output_names = []
    attr = {}  
    try:
        for field, value in fields:
            if _is_in(op_type_case, field.name):
                input_names.append(value)
            elif _is_out(op_type_case, field.name):
                output_names.append('{}/{}'.format(op_conf.name, value))
            else:
                takeit, attr_name, attr_value = _of_field2onnx_attr(onnx_op_type, field, value)
                if takeit:
                    attr[attr_name] = attr_value
    except ValueError as e:
        raise SerializeToJsonError('Failed to serialize {0} field: {1}.'.format(field.name, e))

    return helper.make_node(onnx_op_type, input_names, output_names, name=op_conf.name, **attr)

def Prepare4OnnxGraph(model_load_dir):
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
            #TODO: cur_job_lbn_2_output_remote_blob can be remove, take in's dtype and shape 
            assert lbn in cur_job_lbn_2_output_remote_blob
            dtype = OF_TO_ONNX_DTYPE[cur_job_lbn_2_output_remote_blob[lbn].dtype]
            shape = cur_job_lbn_2_output_remote_blob[lbn].shape 
            lbn = getattr(op_type_pb2, 'in') #use in as the return name
            outputs.append(helper.make_tensor_value_info(lbn, dtype, shape))
        elif op_type_case == 'variable_conf':
            lbn = op_conf.name + '/' + op_type_pb2.out
            dtype = OF_TO_ONNX_DTYPE[op_type_pb2.data_type]
            shape = op_type_pb2.shape.dim
            path = os.path.join(model_load_dir, op_conf.name, op_type_pb2.out)
            assert os.path.isfile(path)
            weight = np.fromfile(path, dtype=OF_TO_NUMPY_DTYPE[op_type_pb2.data_type])
            initializers.append(helper.make_tensor(lbn, dtype, shape, weight))
        else:
            nodes.append(_op_conf2onnx_node(op_conf))
    return nodes, inputs, outputs, initializers


@oneflow_export('export_onnx')
def SaveOnnxModelProto(model_load_dir, save_path='model.onnx', as_text=False, save_readable=True):
    #TODO another option, load model from memory
    opset_id = OperatorSetIdProto()
    opset_id.domain = ''
    opset_id.version = 11 
    global cur_job_lbn_2_output_remote_blob
    job_name = ''
    for key, v in cur_job_lbn_2_output_remote_blob.items():
        job_name = v.job_name_
        break

    nodes, inputs, outputs, initializers = Prepare4OnnxGraph(model_load_dir)
    graph_pb2 = helper.make_graph(nodes, job_name, inputs, outputs, initializer=initializers)
    model = helper.make_model(graph_pb2, ir_version=4, opset_imports=[opset_id],
                              producer_name='oneflow')
    checker.check_model(model)
    save_protobuf(save_path, model, as_text)
    
    if save_readable:
        txt = helper.printable_graph(model.graph)
        with open(save_path+'.graph', 'w') as f:
            f.write(txt)


global cur_job_op_confs
cur_job_op_confs = []

global cur_job_lbn_2_output_remote_blob
cur_job_lbn_2_output_remote_blob = {} 