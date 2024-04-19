# TFLite Conversion

<h3 style="color:#ac5353;"> PyTorch -> ONNX -> TF -> TFLite </h3>

Convert PyTorch Models to TFLite and run inference in TFLite Python API.

## PyTorch to ONNX

Go to [onnx/models](https://github.com/onnx/models) to find the target model's `turnkey_stats.yaml` file. 

Install python environment according to `Python Packages` in the yaml.

```yaml

builtin_model_script: https://github.com/onnx/turnkeyml/blob/main/models/timm/resnet18.py
class: ResNet

system_info:
  Memory Info: 62.79 GB
  OEM System: Virtual Machine
  OS Version: Linux-5.15.0-1051-azure-x86_64-with-glibc2.17
  Processor: Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz
  Python Packages:
  - timm==0.9.8
  - tokenizers==0.14.1
  - toml==0.10.2
  - toolz==0.12.0
  - torch==2.1.0
  - torch-geometric==2.4.0
  - torchaudio==2.1.0
  - torchvision==0.16.0
  - tornado==6.3.3
task: Computer_Vision
```
Download `builtin_model_script` given by the yaml file. Use turnkey to run the .py file.
```bash
turnkey <model>.py
```


## ONNX to TF

Use onnx2tf to implement the conversion. Follow the guidance in [installation.sh](https://github.com/kriskrisliu/PyTorch-ONNX-TFLite/blob/kris/installation/onnx2tf.sh). Then execute the following command:

```bash
onnx2tf -i <model>.onnx  -osd \
--output_integer_quantized_tflite \
--output_folder_path <some-place> 
```

You can find model variants in `output_folder_path`, including:
```bash
resnet18-v1-7_full_integer_quant_with_int16_act.tflite
fingerprint.pb
resnet18-v1-7_integer_quant.tflite
resnet18-v1-7_dynamic_range_quant.tflite
resnet18-v1-7_integer_quant_with_int16_act.tflite
resnet18-v1-7_float16.tflite
saved_model.pb
resnet18-v1-7_float32.tflite
resnet18-v1-7_full_integer_quant.tflite
```

### TFLite Model Inference

For latency:
```python
python scripts/test_latency.py \
-i 100 \
--mode tflite \
-m <model>.tflite;
```

For accuracy:
```python
python scripts/test_imagenet_acc.py \
--mode tflite \
-m <model>.tflite \
-d <some-place>/ImageNet_2012_DataSets/val/
```

### TODO


## References

* [TFLite Documentation](https://www.tensorflow.org/lite/guide)
* [TFLiteConverter TF 1.x version](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/python_api.md)
* [ONNX-TensorFlow](https://github.com/onnx/onnx-tensorflow)
