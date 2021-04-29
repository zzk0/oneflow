import numpy as np

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


if __name__ == "__main__":
    # resnet50 = lambda: nn.Linear(3 * 224 * 224, 100)

    def gf(*args, **kwargs):
        if config.consistent:
            return flow.global_function(*args, **kwargs)
        else:
            return lambda x: x

    def push(name):
        if test_oneflow:
            flow.profiler.range_push("full")
        else:
            pass

    def pop():
        if test_oneflow:
            flow.profiler.range_pop()
        else:
            pass

    @gf()
    def job():
        m = resnet50()
        m = to_cuda(m)
        m.eval()
        with torch.no_grad():
            x = to_cuda(torch.Tensor(np.ones((16, 3 * 224 * 224))))
            y = m(x)
            config.warming = False
            start = time.time()
            push("full")
            for _ in range(10):
                y = m(x)
            pop()
            end = time.time()
            print(end - start)
            print(config.pytime)

    job()
