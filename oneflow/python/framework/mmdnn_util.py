from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, TensorShape, AttrValue

def assign_attr_value(attr, val):
    '''Assign value to AttrValue proto according to data type.'''
    if isinstance(val, bool):
        attr.b = val
    elif isinstance(val, six.integer_types):
        attr.i = val
    elif isinstance(val, float):
        attr.f = val
    elif isinstance(val, str):
        attr.s = val.encode('utf-8')
    elif isinstance(val, TensorShape):
        attr.shape.MergeFromString(val.SerializeToString())
    elif isinstance(val, list):
        if len(val) == 0: return

        if isinstance(val[0], six.integer_types):
            attr.list.i.extend(val)
        elif isinstance(val[0], TensorShape):
            attr.list.shape.extend(val)
        else:
            raise NotImplementedError('AttrValue cannot be of %s %s' % (type(val), type(val[0])))
    else:
        raise NotImplementedError('AttrValue cannot be of %s' % type(val))

def _input_name_of2mm(name):
    return name[:name.rfind('/')]

def CreateIRNode(op_conf):
    node_pb2 = NodeDef()
    node_pb2.name = op_conf.name
    op_type_case = op_conf.WhichOneof('op_type')
    node_pb2.op = op_type_case
    op_type_pb2 = getattr(op_conf, op_type_case)

    #if op_type_pb2.HasField('in'):
    if 'in' in dir(op_type_pb2):
        in_value = getattr(op_type_pb2, 'in')
        if isinstance(in_value, str):
            if in_value != "":
                node_pb2.input.append(_input_name_of2mm(in_value))
        else: #<class 'google.protobuf.internal.containers.RepeatedScalarFieldContainer'>
            for v in in_value:
                node_pb2.input.append(_input_name_of2mm(v))
    global cur_job_user_op_name2ir_node
    cur_job_user_op_name2ir_node[op_conf.name] = node_pb2

def ClearIRNodes():
    global cur_job_user_op_name2ir_node
    cur_job_user_op_name2ir_node = {}


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


def SaveIRGraphDef(path='mmdnn_graph'):
    graph_pb2 = GraphDef()
    graph_pb2.version = 0
    graph_pb2.node.extend([node for node in cur_job_user_op_name2ir_node.values()])
    _save_to_json(graph_pb2, path+'.json')
    _save_to_proto(graph_pb2, path+'.pb')


global cur_job_user_op_name2ir_node
cur_job_user_op_name2ir_node = {}
