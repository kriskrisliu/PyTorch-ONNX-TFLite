#!/bin/bash

# Define an array of model names
declare -a ONNX_MODELS=("resnet18-v1-7" "resnet50-v1-7" "ssd-10" "vgg16-12" "resnet101-v1-7" "resnet34-v1-7" "ssd-12" "vgg19-7")

# Loop through each model name in the array
for ONNX_MODEL in "${ONNX_MODELS[@]}"
do
    # Set the output path
    OUTPUT_PATH="output/${ONNX_MODEL}"

    # Execute the conversion command
    onnx2tf -i "model_zoo/${ONNX_MODEL}.onnx" -osd --output_integer_quantized_tflite --output_folder_path "$OUTPUT_PATH"
done


# inference for acc
python scripts/test_imagenet_acc.py -mode tflite -m output/resnet18-v1-7/resnet18-v1-7_float32.tflite -d /data-hdd/ImageNet_2012_DataSets/val/

# torch to onnx
turnkey resnet18.py