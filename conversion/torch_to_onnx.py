import torch
from torchvision.models import mobilenet_v2
from timm import list_models
import timm
import torch

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Export a model to ONNX format.")

# Add arguments
parser.add_argument('--show_list', action='store_true', help='Show a list of available models that can be exported.')
parser.add_argument('--model_name', type=str, default='mobilenet_v2', help='Name of the model to export. Default is "mobilenet_v2".')
parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity for debugging.')
parser.add_argument('--img_size', type=int, nargs="+", required=True, help='Expected image size for the model input, e.g., --img_size 224 224')
parser.add_argument('--opset_version', type=int, default=14, help='ONNX opset version to use for exporting the model. Default is 14.')

# Parse arguments
args = parser.parse_args()

# Example usage and implementation
if args.show_list:
    # Assuming 'list_models' and 'timm' are already imported and available
    model_zoo = timm.list_models(pretrained=True)
    print("Available pretrained models:")
    for model in model_zoo:
        print(model)


model_name = args.model_name
model = timm.create_model(f"{model_name}", pretrained=True)
model.eval()
if args.verbose:
    print(model)

img_size = args.img_size
batch_size = 1
onnx_model_path = f"{model_name}.onnx"

sample_input = torch.rand((batch_size, 3, *img_size))

y = model(sample_input)

torch.onnx.export(
    model,
    sample_input, 
    onnx_model_path,
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=args.opset_version
)