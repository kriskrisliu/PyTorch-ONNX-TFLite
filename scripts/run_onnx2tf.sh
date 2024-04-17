#!/bin/bash

# Define an array of model names
declare -a ONNX_MODELS=("resnet50-v1-7" "ssd-10" "vgg16-12" "resnet101-v1-7" "resnet34-v1-7" "ssd-12" "vgg19-7")

# Loop through each model name in the array
for ONNX_MODEL in "${ONNX_MODELS[@]}"
do
    # Set the output path
    OUTPUT_PATH="output/${ONNX_MODEL}"

    # Execute the conversion command
    onnx2tf -i "model_zoo/${ONNX_MODEL}.onnx" -osd --output_integer_quantized_tflite --output_folder_path "$OUTPUT_PATH"
done
