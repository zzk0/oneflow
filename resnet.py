import numpy as np
from pyinstrument import Profiler
import cProfile
from line_profiler import LineProfiler

import time
import sys
import config

test_oneflow = len(sys.argv) == 1
if test_oneflow:
    import oneflow as flow
    import oneflow.nn as nn
    import resnet_model

    torch = flow
    flow.enable_eager_execution()
    resnet50 = resnet_model.resnet50
else:
    import torch
    import torch.nn as nn
    import torchvision.models as models

    resnet50 = models.resnet50


def to_cuda(x):
    if test_oneflow:
        return x
    else:
        return x.to("cuda")


def sync(x):
    if test_oneflow:
        return x.numpy()
    else:
        return x.cpu().numpy()


def get_tensor(shape):
    return torch.Tensor(np.ones(shape))
    op = (
        flow.builtin_op("constant")
        .Output("out")
        .Attr("is_floating_value", True)
        .Attr("floating_value", 3.0)
        .Attr("dtype", flow.float32)
        .Attr("shape", shape)
        .Build()
    )
    return op()[0]


if __name__ == "__main__":
    # resnet50 = lambda: nn.Linear(3 * 224 * 224, 100)
    # shape = (16, 3 * 224 * 224)
    # times = 8000

    shape = (16, 3, 224, 224)
    times = 50
    # times = 300

    def gf(*args, **kwargs):
        if config.consistent:
            return flow.global_function(*args, **kwargs)
        else:
            return lambda x: x

    def push(name):
        if test_oneflow and not config.warming:
            flow.profiler.range_push(name)
        else:
            pass

    def pop():
        if test_oneflow and not config.warming:
            flow.profiler.range_pop()
        else:
            pass

    m = resnet50()
    m = to_cuda(m)
    m.eval()
    # for x in m.parameters():
    #     x.determine()
    # for x in m.buffers():
    #     x.determine()

    def warmup():
        with torch.no_grad():
            x = to_cuda(get_tensor(shape))
            for _ in range(5):
                y = m(x)
                sync(y)

    warmup()

    @gf()
    def run():
        with torch.no_grad():
            x = to_cuda(get_tensor(shape))
            config.warming = False
            print("sleeping 5s..")
            time.sleep(5)
            print("sleeping finish")
            start = time.time()
            # profiler = Profiler()
            # profiler.start()
            # pr = cProfile.Profile()
            # pr.enable()
            for _ in range(times):
                # push("full")
                y = m(x)
                # pop()
                sync(y)
            # pr.disable()
            # pr.print_stats()

            # profiler.stop()
            # print(profiler.output_text(unicode=True, color=True, show_all=True))

            end = time.time()
            total_time = end - start
            one_time = total_time / times
            print(f"time: {total_time} / {times} = {one_time}")
            # print(config.pytime)

    run()
