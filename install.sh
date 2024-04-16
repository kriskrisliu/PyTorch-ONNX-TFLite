conda create -n samsung-tflite python=3.7 -y

# --trusted-host mirrors.aliyun.com
# torch 1.7.1 cu110
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# timm 0.3.2
pip install timm==0.3.2 tensorflow-cpu==2.4.1 \
onnx==1.8.0 onnx-tf==1.7.0 typeguard==2.8.0 \
typing_extensions==3.7.4.3 \
tensorflow-addons==0.12.0 keras==2.11

# pip install tensorflow-cpu==2.4.1

# pip install onnx==1.8.0 onnx-tf==1.7.0

# pip install typeguard==2.8.0 typing_extensions==3.7.4.3

# pip install tensorflow-addons==0.12.0

# pip install keras==2.11