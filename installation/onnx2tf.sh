conda create -n samsung-onnx2tf python=3.10 -y
conda activate samsung-onnx2tf
pip install onnx==1.15.0
pip install git+https://github.com/PINTO0309/onnx2tf.git@1.20.0

pip install tensorflow-cpu==2.16.1

pip install onnxruntime==1.17.1 onnx-simplifier==0.4.33 onnx_graphsurgeon simple_onnx_processing_tools psutil==5.9.5 ml_dtypes==0.3.2
pip install -U tf-keras~=2.16


# run

# ONNX_MODEL="model_zoo/resnet18-v1-7.onnx";\
# OUTPUT_PATH="output/resnet18";\
# onnx2tf -i $ONNX_MODEL -osd --output_integer_quantized_tflite --output_folder_path $OUTPUT_PATH



#!/bin/bash

# Define an array of model names
# declare -a ONNX_MODELS=("resnet18-v1-7" "resnet50-v1-7" "ssd-10" "vgg16-12" "resnet101-v1-7" "resnet34-v1-7" "ssd-12" "vgg19-7")

# # Loop through each model name in the array
# for ONNX_MODEL in "${ONNX_MODELS[@]}"
# do
#     # Set the output path
#     OUTPUT_PATH="output/${ONNX_MODEL}"

#     # Execute the conversion command
#     onnx2tf -i "model_zoo/${ONNX_MODEL}.onnx" -osd --output_integer_quantized_tflite --output_folder_path "$OUTPUT_PATH"
# done
