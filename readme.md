# ncnn推理测试

本项目进行ncnn的推理测试

## 编译

```shell
mkdir build
cd build
cmake ..
make
```

## 执行推理

执行脚本参考[`scripts/run_mobilenerssd.sh`](scripts/run_mobilenerssd.sh)文件

参数说明如下

```shell
ncnn_test <ncnn_model_bin_path> <ncnn_model_param_path> <input_layer_name> <output_layer_name> <npy_data_path> <result_save_path> <print_flag>
```

参数中文说明如下

```shell
ncnn_test <ncnn模型bin文件路径> <ncnn模型bin文件路径> <模型输入层名称> <模型输出层名称> <输入数据npy文件路径> <输出保存路径> <开启输出打印>
```

以下是`scripts/run_mobilenerssd.sh`文件中的命令示例

```shell
../build/ncnn_test ../samples/mobilenetssd.bin ../samples/mobilenetssd.param data detection_out ../samples/data.npy ../samples/feature 1
```

## npy文件生成

npy文件可使用[`scripts/image2npy.py`](scripts/image2npy.py)获取