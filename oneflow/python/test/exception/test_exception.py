"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import oneflow as flow
import oneflow_api
from oneflow_api.exception import (
    ErrorException,
    TodoException,
    UnimplementedException,
    OneflowException,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--type", type=str, default="todo", choices=["todo", "unimpl", "error"]
)
parser.add_argument("-r", "--raise_exp", type=str, default="n")
args = parser.parse_args()


def TestOneflowError():
    try:
        oneflow_api.TestErrorException(args.type)
    except TodoException:
        print("Receive TodoException!")
        if args.raise_exp == "y":
            raise
    except UnimplementedException:
        print("Receive UnimplementedException!")
        if args.raise_exp == "y":
            raise
    except ErrorException:
        print("Receive ErrorException!")
        if args.raise_exp == "y":
            raise


if __name__ == "__main__":
    TestOneflowError()
