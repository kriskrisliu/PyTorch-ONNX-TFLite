from timm import list_models
import timm
import torch
from onnx_tf.backend import prepare
import onnx


model_zoo = list_models(pretrained=True)
model_name = 'vit'
for name in model_zoo:
    if model_name in name:
        print(name)

model_name = "vit_tiny_patch16_224.augreg_in21k"
model = timm.create_model(f"hf_hub:timm/{model_name}", pretrained=True)
print(model)

img_size = (224, 224)
batch_size = 1
sample_input = torch.rand((batch_size, 3, *img_size))
onnx_model_path = f"{model_name}.onnx"

torch.onnx.export(
    model,                  # PyTorch Model
    sample_input,                    # Input tensor
    onnx_model_path,        # Output file (eg. 'output_model.onnx')
    opset_version=14,       # Operator support version
    input_names=['input'],   # Input tensor name (arbitary)
    output_names=['output'] # Output tensor name (arbitary)
)


tf_model_path = f'{model_name}.pb'

onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)