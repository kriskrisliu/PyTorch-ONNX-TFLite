# convert pytorch to onnx
# conda activate samsung-torch2onnx
python conversion/torch_to_onnx.py --show_list --model_name $1 -v --img_size $2 $3 --opset_version $4
# python conversion/torch_to_onnx.py --show_list --model_name vit_small_patch16_224 -v --img_size 224 224 --opset_version 12
python conversion/onnx_to_tf.py
python conversion/tf_to_tflite.py
# conda deactivate



