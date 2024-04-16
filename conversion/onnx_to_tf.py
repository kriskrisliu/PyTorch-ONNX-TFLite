from onnx_tf.backend import prepare
import onnx

# onnx_model_path = 'vit_small_patch16_224.onnx'
# tf_model_path = 'vit_small_patch16_224'
onnx_model_path = 'resnet50.onnx'
tf_model_path = 'resnet50_cpu'

onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model, device='CPU')
tf_rep.export_graph(tf_model_path)