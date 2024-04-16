import numpy as np
import tensorflow as tf
import os
import time
from tqdm import tqdm
import argparse  # Import argparse for command-line options

# Setup command line arguments
parser = argparse.ArgumentParser(description='Test TFLite model inference latency.')
parser.add_argument('-i','--iterations', type=int, default=10,
                    help='Number of times to run the inference for averaging')
parser.add_argument('-m','--model_path', type=str, default='vit_base_patch16_224.tflite',
                    help='Path to the TFLite model file')
parser.add_argument('--mode', type=str, default='tflite',
                    help='tensorflow or tflite')
args = parser.parse_args()

# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if args.mode == "tflite":
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Setup to test inference time
    input_shape = input_details[0]['shape']
    print(input_shape)
    iterations = args.iterations  # Use the passed number of iterations
    latency_list = []
    pbar = tqdm(range(iterations), desc="Testing Latency")

    for _ in pbar:
        pbar.set_postfix({
            "shape":input_shape,
        })

        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        start_time = time.time()  # Start time
        interpreter.invoke()
        end_time = time.time()  # End time
        
        latency = end_time - start_time
        latency_list.append(latency)

    # Calculate average latency after removing the first result (warm-up)
    latency_list = latency_list[1:]  # Skip the first measurement if it's a warm-up
    average_latency = sum(latency_list) / len(latency_list)
    print("[TFlite] Average inference latency: {:.4f} seconds".format(average_latency))
elif args.mode == "tensorflow":
    # Load TensorFlow model
    model = tf.saved_model.load(args.model_path)
    for key, value in model.signatures.items():
        for input_tensor in value.inputs:
            input_shape = input_tensor.shape
            break
        break

    model.trainable = False

    # Setup to test inference time
    latency_list = []
    pbar = tqdm(range(args.iterations), desc="Testing Latency")

    for _ in pbar:
        input_tensor = tf.random.uniform(input_shape)

        start_time = time.time()  # Start time
        out = model(**{'input': input_tensor})  # Assuming the model takes a single input tensor
        end_time = time.time()  # End time
        
        latency = end_time - start_time
        latency_list.append(latency)

    # Calculate average latency after removing the first result (warm-up)
    latency_list = latency_list[1:]  # Skip the first measurement if it's a warm-up
    average_latency = sum(latency_list) / len(latency_list)
    print("[TensorFlow] Average inference latency: {:.4f} seconds".format(average_latency))