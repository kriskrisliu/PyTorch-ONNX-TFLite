import numpy as np
import tensorflow as tf
import os
import time
from tqdm import tqdm
import argparse  # Import argparse for command-line options
import numpy as np
from PIL import Image
import scipy
import multiprocessing

def args_parser():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Test TFLite model inference latency.')
    parser.add_argument('-m','--model_path', type=str, default='vit_base_patch16_224.tflite',
                        help='Path to the TFLite model file')
    parser.add_argument('--mode', type=str, default='tflite',
                        help='tensorflow or tflite')
    parser.add_argument('-d','--val_path', type=str, default='/data-hdd/ImageNet_2012_DataSets/val/',
                        help='Path to the imagenet val')
    args = parser.parse_args()
    return args

def image_resize(image, min_len):
    image = Image.fromarray(image)
    ratio = float(min_len) / min(image.size[0], image.size[1])
    if image.size[0] > image.size[1]:
        new_size = (int(round(ratio * image.size[0])), min_len)
    else:
        new_size = (min_len, int(round(ratio * image.size[1])))
    image = image.resize(new_size, Image.BILINEAR)
    return np.array(image)

def crop_center(image, crop_w, crop_h):
    h, w, c = image.shape
    start_x = w//2 - crop_w//2
    start_y = h//2 - crop_h//2
    return image[start_y:start_y+crop_h, start_x:start_x+crop_w, :]

def preprocess(image):
    # resize so that the shorter side is 256, maintaining aspect ratio
    image = Image.open(image)
    image = image.convert('RGB')
    image = np.array(image)
    # print(image.shape)

    image = image_resize(image, 256)

    # Crop centered window 224x224
    image = crop_center(image, 224, 224)

    # transpose
    image = image.transpose(2, 0, 1)

    # convert the input data into the float32 input
    img_data = image.astype('float32')

    # normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

    # add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    norm_img_data = norm_img_data.transpose(0, 2, 3, 1)
    return norm_img_data

def val_one_class(img_folder, gt, args, verbose=False):
    preds = []
    path_list = [os.path.join(img_folder,im) for im in os.listdir(img_folder)]
    for img_path in path_list:
        norm_img_data = preprocess(img_path)
        # print(norm_img_data.shape)

        if args.mode == "tflite":
            # Load the TFLite model and allocate tensors
            interpreter = tf.lite.Interpreter(model_path=args.model_path)
            interpreter.allocate_tensors()

            # Get input and output tensors
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Setup to test inference time
            input_shape = input_details[0]['shape']
            # print(input_shape)

            input_data = norm_img_data
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            # print(output_data)
            # print(output_data.shape)
            pred_id = scipy.special.softmax(output_data).argmax()
            if verbose: print(gt,"|",labels[pred_id])
            correct = (gt == labels[pred_id].split(" ")[0])
            preds += [correct]
            # import ipdb;ipdb.set_trace()

        elif args.mode == "tensorflow":
            raise NotImplementedError
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
    print(f"Finish: {img_folder}")
    return preds

def evaluate_class(data):
    args, gt = data  # Unpack the tuple here
    img_folder = os.path.join(args.val_path, gt)
    return val_one_class(img_folder, gt, args)

# if __name__=="__main__":
#     with open('scripts/synset.txt', 'r') as f:
#         labels = [l.rstrip() for l in f]
#     # print(labels)
#      # disable GPU
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#     args = args_parser()
#     gts = os.listdir(args.val_path)
#     preds = []
#     pbar = tqdm(gts, desc="Evaluate ImageNet")
#     for gt in pbar:
#         img_folder = os.path.join(args.val_path, gt) #pick
#         preds_per_class = val_one_class(img_folder, gt, args)
#         preds += preds_per_class
#         pbar.set_postfix({
#             "Acc": sum(preds)/len(preds)
#         })
    
#     print(f"{sum(preds)}/{len(preds)}, {(sum(preds)/len(preds)*100):.2f} %")
if __name__=="__main__":
    # Disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    args = args_parser()

    with open('scripts/synset.txt', 'r') as f:
        labels = [l.rstrip() for l in f]

    gts = os.listdir(args.val_path)
    preds = []

    # Set up multiprocessing pool
    print(f"Spawn {multiprocessing.cpu_count()//4} processes among totally {multiprocessing.cpu_count()}!")
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()//4)

    # Distribute the workload
    results = pool.map(evaluate_class, [(args, gt) for gt in gts])

    # Flatten results from all classes
    preds = [pred for sublist in results for pred in sublist]

    # Compute the accuracy
    accuracy = sum(preds) / len(preds)
    print(f"{sum(preds)}/{len(preds)}, {accuracy * 100:.2f} %")
