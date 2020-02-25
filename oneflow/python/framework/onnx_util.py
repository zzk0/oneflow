from google.protobuf import descriptor
from onnx import ModelProto, GraphProto, NodeProto, OperatorSetIdProto
from onnx import helper, onnx_pb
from oneflow.python.oneflow_export import oneflow_export


def _io_name_of2onnx(io_name):
    if in_or_out == 'out':
        return '{}/{}'.format(op_name, io_name)
    return io_name
    #return name[:name.rfind('/')]

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

def CreateIRNode(op_conf):
    op_type, inputs, outputs, attr = PrepareOnnxArgs(op_conf)
    print(op_type, inputs, outputs, attr)
    node_pb2 = helper.make_node(op_type, inputs, outputs, name=op_conf.name, **attr)

    global cur_job_user_op_name2onnx_node
    cur_job_user_op_name2onnx_node[op_conf.name] = node_pb2

def ClearIRNodes():
    global cur_job_user_op_name2onnx_node
    cur_job_user_op_name2onnx_node = {}


@oneflow_export('export_onnx')
def SaveOnnxModelProto(path='model.onnx'):
    opset_id = OperatorSetIdProto()
    opset_id.domain = ''
    opset_id.version = 9

    nodes = [node for node in cur_job_user_op_name2onnx_node.values()]
    graph_pb2 = helper.make_graph(nodes,
                                  'oneflow-export', #TODO get func name,
                                  [],#TODO input
                                  [],#TODO output
                                  initializer=[], #TODO feed weight here
                                 )
    model = helper.make_model(graph_pb2, ir_version=4, opset_imports=[opset_id],
                              producer_name='oneflow')
    return model


global cur_job_user_op_name2onnx_node
cur_job_user_op_name2onnx_node = {}
