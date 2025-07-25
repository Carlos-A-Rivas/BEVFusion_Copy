#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# export CUDA_VISIBLE_DEVICES=2

export TensorRT_Lib=/usr/lib/x86_64-linux-gnu
export TensorRT_Inc=/usr/include/x86_64-linux-gnu
export TensorRT_Bin=/usr/src/tensorrt/bin

export CUDA_Lib=/usr/local/cuda/lib64
export CUDA_Inc=/usr/local/cuda/include
export CUDA_Bin=/usr/local/cuda-12.2/bin
export CUDA_HOME=/usr/local/cuda-12.2

export CUDNN_Lib=/usr/lib/x86_64-linux-gnu

# export TensorRT_Lib=${TENSORRT_LIB}
# export TensorRT_Inc=${TENSORRT_INCLUDE}
# export TensorRT_Bin=${TENSORRT_BIN}

# export CUDA_Lib=${CUDA_LIB}
# export CUDA_Inc=${CUDA_INCLUDE}
# export CUDA_Bin=${CUDA_BIN}
# export CUDA_HOME=/path/to/cuda

# export CUDNN_Lib=${CUDNN_LIB}

# Just used to distinguish the libspconv version, it doesn't affect the version of cuda used by your application
# For CUDA-11.x:    SPCONV_CUDA_VERSION=11.4
# For CUDA-12.x:    SPCONV_CUDA_VERSION=12.6
export SPCONV_CUDA_VERSION=12.8

# resnet50/resnet50int8/swint
export DEBUG_MODEL=resnet50int8

# fp16/int8
export DEBUG_PRECISION=int8
export DEBUG_DATA=custom-example
export USE_Python=OFF

# check the configuration path
# clean the configuration status
export ConfigurationStatus=Failed
if [ ! -f "${TensorRT_Bin}/trtexec" ]; then
    echo "Can not find ${TensorRT_Bin}/trtexec, there may be a mistake in the directory you configured."
    return
fi

if [ ! -f "${CUDA_Bin}/nvcc" ]; then
    echo "Can not find ${CUDA_Bin}/nvcc, there may be a mistake in the directory you configured."
    return
fi

echo "=========================================================="
echo "||  MODEL: $DEBUG_MODEL"
echo "||  PRECISION: $DEBUG_PRECISION"
echo "||  DATA: $DEBUG_DATA"
echo "||  USEPython: $USE_Python"
echo "||"
echo "||  TensorRT: $TensorRT_Lib"
echo "||  CUDA: $CUDA_HOME"
echo "||  CUDNN: $CUDNN_Lib"
echo "=========================================================="

BuildDirectory=`pwd`/build

if [ "$USE_Python" == "ON" ]; then
    export Python_Inc=`python3 -c "import sysconfig;print(sysconfig.get_path('include'))"`
    export Python_Lib=`python3 -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))"`
    export Python_Soname=`python3 -c "import sysconfig;import re;print(re.sub('.a', '.so', sysconfig.get_config_var('LIBRARY')))"`
    echo Find Python_Inc: $Python_Inc
    echo Find Python_Lib: $Python_Lib
    echo Find Python_Soname: $Python_Soname
fi

export PATH=$TensorRT_Bin:$CUDA_Bin:$PATH
export LD_LIBRARY_PATH=$TensorRT_Lib:$CUDA_Lib:$CUDNN_Lib:$BuildDirectory:$LD_LIBRARY_PATH
export PYTHONPATH=$BuildDirectory:$PYTHONPATH
export ConfigurationStatus=Success

if [ -f "tool/cudasm.sh" ]; then
    echo "Try to get the current device SM"
    . "tool/cudasm.sh"
    echo "Current CUDA SM: $cudasm"
fi

export CUDASM=$cudasm

echo Configuration done!