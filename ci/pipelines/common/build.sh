set -ex
export pkg_type=${ONEFLOW_PKG_TYPE:-"cuda"}
export tmp_dir=${ONEFLOW_PKG_TMP_DIR:-"$HOME/oneflow-pkg/${ONEFLOW_PKG_TYPE}"}
export wheelhouse_dir=$ci_tmp_dir/wheelhouse
export src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}

cuda_version=10.2
extr_args="-DFOR_CI=ON"
if [[ ${ONEFLOW_PKG_TYPE} -eq xla ]]
then
    extr_args="${extr_args} --xla"
    cuda_version="10.1"
fi
if [[ ${ONEFLOW_PKG_TYPE} -eq cpu ]]
then
    extr_args="${extr_args} --cpu"
fi
mkdir -p ${tmp_dir}
cd ${tmp_dir}
docker run --rm -v $PWD:/p -w $PWD:/p busybox rm -rf /p/wheelhouse
python3 ${src_dir}/docker/package/manylinux/build_wheel.py \
    --cuda_version=${cuda_version} \
    --python_version=3.6 \
    --use_tuna --use_system_proxy --use_aliyun_mirror \
    --wheel_house_dir=${wheelhouse_dir} \
    --oneflow_src_dir=${src_dir} $extr_args
