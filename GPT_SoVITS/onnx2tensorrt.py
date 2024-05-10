import os
import shutil
import tensorrt as trt
import onnx
import glob
import numpy as np
import sys

# from onnx_graphsurgeon import GraphSurgeon
TRT_LOGGER = trt.Logger()


def make_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.mkdir(dir)
    else:
        os.mkdir(dir)


def get_onnx_inputs(onnx_model_path):
    # 加载ONNX模型
    model = onnx.load(onnx_model_path)

    # 获取所有输入张量的名称和形状
    input_tensors = []
    for input in model.graph.input:
        input_name = input.name
        input_shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        input_tensors.append((input_name, input_shape))

    # 打印所有输入张量的名称和形状
    for input_tensor in input_tensors:
        print("Input Tensor Name: {}, Shape: {}".format(input_tensor[0], input_tensor[1]))

    return input_tensors


def onnx2tensorrt(onnx_model_path, input_shapes, rt_folder=None):
    if rt_folder is None:
        onnx_engine_path = onnx_model_path.replace('.onnx', '.trt')
    else:
        rt_f = os.path.basename(onnx_model_path).replace('.onnx', '.trt')
        onnx_engine_path = os.path.join(rt_folder, rt_f)

    # ONNX_build_engine(onnx_model_path, onnx_engine_path, True)
    # 创建TensorRT构建器
    builder = trt.Builder(trt.Logger(trt.Logger.INFO))

    # 1、动态输入第一点必须要写的
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    batch_size = 8  # trt推理时最大支持的batchsize
    # 创建TensorRT网络
    network = builder.create_network(explicit_batch)

    # 将ONNX模型转换为TensorRT网络
    parser = trt.OnnxParser(network, trt.Logger(trt.Logger.INFO))
    # 加载ONNX模型并解析
    with open(onnx_model_path, 'rb') as model:
        parsed = parser.parse(model.read())

    # 检查解析是否成功
    if not parsed:
        for error in range(parser.num_errors):
            print(parser.get_error(error))
            return

    # 配置和构建TensorRT引擎
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()  # 动态输入时候需要
    for key in input_shapes.keys():
        # 重点
        # 分别为最小输入、常规输入、最大输入
        # 有几个输入就要写几个profile.set_shape 名字和转onnx的时候要对应
        # tensorrt6以后的版本是支持动态输入的，需要给每个动态输入绑定一个profile，用于指定最小值，常规值和最大值，如果超出这个范围会报异常。
        shape = input_shapes[key]
        profile.set_shape(str(key), shape[0], shape[1], shape[2])
        config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    # 保存TensorRT引擎
    if engine_bytes is None:
        print("Failed to create engine")
        sys.exit(1)

    with open(onnx_engine_path, "wb") as f:
        print("Serializing engine to file: {:}".format(onnx_engine_path))
        f.write(engine_bytes)


def main():
    onnx_folder = 'onnx/nahida'
    onnxs = glob.glob(onnx_folder + "/*.onnx")

    rt_folder = 'tensorrt_model/nahida'
    make_dir(rt_folder)

    shapes = {
        "nahida_t2s_encoder_shapes": {"ref_seq": [(1, 1), (1, 256), (1, 1024)],
                                      "text_seq": [(1, 1), (1, 256), (1, 1024)],
                                      "ref_bert": [(1, 1024), (256, 1024), (1024, 1024)],
                                      "text_bert": [(1, 1024), (256, 1024), (1024, 1024)],
                                      "ssl_content": [(1, 768, 1), (1, 768, 256), (1, 768, 1024)]},
        "nahida_t2s_fsdec_shapes": {"x": [(1, 1, 512), (1, 256, 512), (1, 1024, 512)],
                                    "prompts": [(1, 1), (1, 256), (1, 1024)]},
        "nahida_t2s_sdec_shapes": {"iy": [(1, 1), (1, 256), (1, 1024)],
                                   "ik": [(24, 1, 1, 512), (24, 256, 1, 512), (24, 1024, 1, 512)],
                                   "iv": [(24, 1, 1, 512), (24, 256, 1, 512), (24, 1024, 1, 512)],
                                   "iy_emb": [(1, 1, 512), (1, 256, 512), (1, 1024, 512)],
                                   "ix_example": [(1, 1), (1, 256), (1, 1024)]},
        "nahida_vits_shapes": {"text_seq": [(1, 1), (1, 256), (1, 1024)],
                               "pred_semantic": [(1, 1, 1), (1, 1, 256), (1, 1, 1024)],
                               "ref_audio": [(1, 1), (1, 256), (1, 1024)]}}
    # for f in onnxs:
    #     print(f)
    #     get_onnx_inputs(f)

    for f in onnxs:
        shape_name = os.path.basename(f).split('.')[0] + "_shapes"
        print("== proc {} ==".format(f))
        get_onnx_inputs(f)
        onnx2tensorrt(f, shapes[shape_name], rt_folder)


if __name__ == "__main__":
    # 获取TensorRT的版本信息
    trt_version = trt.__version__
    print("TensorRT版本：", trt_version)

    main()
