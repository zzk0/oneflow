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

def CreateIRNode(op_conf):
    op_type_case = op_conf.WhichOneof('op_type')
    op_type = _of_op_type_case2onnx_op_type(op_type_case)

    op_type_pb2 = getattr(op_conf, op_type_case)

    def append_in_out(in_or_out):
        ret = []
        gen_io_name = lambda n: n if in_or_out=='in' else '{}/{}'.format(op_conf.name, n)
        #if op_type_pb2.HasField('in'):
        if in_or_out in dir(op_type_pb2):
            value = getattr(op_type_pb2, in_or_out)
            if isinstance(value, str):
                if value != "":
                    ret.append(gen_io_name(value))
            else: #<class 'google.protobuf.internal.containers.RepeatedScalarFieldContainer'>
                for v in value:
                    ret.append(gen_io_name(v))
        return ret

    inputs = append_in_out('in')
    outputs = append_in_out('out')

    node_pb2 = helper.make_node(op_type, inputs, outputs, name=op_conf.name)

    global cur_job_user_op_name2onnx_node
    cur_job_user_op_name2onnx_node[op_conf.name] = node_pb2

def ClearIRNodes():
    global cur_job_user_op_name2onnx_node
    cur_job_user_op_name2onnx_node = {}


def _save_to_json(IR_graph, filename):
    import google.protobuf.json_format as json_format
    json_str = json_format.MessageToJson(IR_graph, preserving_proto_field_name = True)

    with open(filename, "w") as of:
        of.write(json_str)

    print ("IR network structure is saved as [{}].".format(filename))

    return json_str


def _save_to_proto(IR_graph, filename):
    proto_str = IR_graph.SerializeToString()
    with open(filename, 'wb') as of:
        of.write(proto_str)

    print ("IR network structure is saved as [{}].".format(filename))

    return proto_str


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
