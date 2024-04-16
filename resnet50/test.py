import keras
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Convert a Keras model to TensorFlow Lite format.')
parser.add_argument('-p','--precision', type=str, default='float32', choices=['int8', 'float32'],
                    help='Precision type for the TensorFlow Lite model (default: float32).')
args = parser.parse_args()

precision = args.precision

model = keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)


tf_callable = tf.function(
      model.call,
      autograph=False,
      input_signature=[tf.TensorSpec((1,224,224,3), tf.float32)],
)
tf_concrete_function = tf_callable.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [tf_concrete_function], tf_callable
)


if precision=="int8":
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    def representative_dataset_gen():
        for _ in range(100):
            data = np.random.rand(1, 224, 224, 3).astype(np.float32)
            #yield {'inputs0': data}
            yield [data]
            
    converter.representative_dataset = representative_dataset_gen

    tflite_model = converter.convert()
elif precision=="float32":
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_FLOAT32]
    converter.optimizations = []
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()
else:
    raise NotImplementedError

f = f"ResNet50_{precision}.tflite"
with open(f, "wb") as fp:
    fp.write(tflite_model)
