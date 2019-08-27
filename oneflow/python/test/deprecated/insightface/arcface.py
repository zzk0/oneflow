

def ArcFace(input_blob, embedding_size=512, num_classes=85742,use_model_parallel=False):
  dl_net = input_blob.dl_net()
  fc1 = input_blob
  easy_margin = False
  s = 64
  m = args.margin_m
  assert s > 0.0
  assert m >= 0.0
  assert m < (math.pi/2)    
  #fc7
  _weight = dl_net.Variable(shape = {'dim':[args.num_classes, args.emb_size]}, model_name='weight', name = "fc7")
  _weight = dl_net.L2Normalize(_weight, name = "L2Normalize_0", axis=1, epsilon = 1e-10)
  nembedding = dl_net.L2Normalize(fc1, name = "L2Normalize_1", axis=1, epsilon = 1e-10)
  nembedding = dl_net.ScalarMul(nembedding, s, name = "ScalarMul_0")
  if args.use_model_parallel:
    nembedding = dl_net.ParallelCast(nembedding, name = "ParallelCast_0", broadcast_parallel = {})
  fc7 = dl_net.Matmul(nembedding, _weight, name = "Matmul_0", transpose_b = True)
  
  if args.use_model_parallel:
    label = dl_net.ParallelCast(label, name = "ParallelCast_1", broadcast_parallel = {})

  label_expand_dim = dl_net.ExpandDims(label, name = "ExpandDims_0", dim = 1)
  zy = dl_net.BatchGather(in_blob = fc7, indices_blob = label_expand_dim, name = "pick", depth= args.num_classes)
  if args.use_model_parallel:
    zy = dl_net.ParallelCast(zy, split_parallel = {'axis' : 0})
  cos_t = dl_net.ScalarMul(zy, scalar = 1.0/s)
  cos_m = math.cos(m)
  sin_m = math.sin(m)
  mm = math.sin(math.pi-m)*m
  threshold=math.cos(math.pi-m)
  if easy_margin:
    cond = dl_net.Relu(cos_t)
  else:
    cond_v = dl_net.ScalarAdd(cos_t, -threshold)
    cond = dl_net.Relu(cond_v)
   body = dl_net.Square(cos_t)
  body = dl_net.ScalarMul(body, -1)
  body = dl_net.ScalarAdd(body, 1.0)
  sin_t = dl_net.Sqrt(body)
  new_zy = dl_net.ScalarMul(cos_t, cos_m)
  b = dl_net.ScalarMul(sin_t, sin_m)
  neg_b = dl_net.ScalarMul(b, -1)
  new_zy = dl_net.Add([new_zy, neg_b], name="Add_20")
  new_zy = dl_net.ScalarMul(new_zy, s)
  if easy_margin:
    zy_keep = zy
  else:
    zy_keep = dl_net.ScalarAdd(zy, -s*mm)
  new_zy = dl_net.Where(cond, new_zy, zy_keep)
  neg_zy = dl_net.ScalarMul(zy, -1)
  diff = dl_net.Add([new_zy, neg_zy], name="Add_21")
  gt_one_hot = dl_net.OneHot(label, args.num_classes, name='OneHot_0')
  if args.use_model_parallel:
    diff = dl_net.ParallelCast(diff, broadcast_parallel = {})
  body = dl_net.BroadcastMul(gt_one_hot, diff)
  fc7 = dl_net.Add([fc7, body], name="Add_22")
  return fc7