import tensorflow as tf

# saved_model_dir = 'vit_small_patch16_224'
# tflite_model_path = 'vit_small_patch16_224_float32.tflite'
saved_model_dir = 'resnet50_cpu'
tflite_model_path = 'resnet50_cpu.tflite'

## TFLite Conversion
# Before conversion, fix the model input size
model = tf.saved_model.load(saved_model_dir)
model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[0].set_shape([1, 3, 224, 224])
tf.saved_model.save(model, "saved_model_updated", signatures=model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY])

# Convert
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir='saved_model_updated', signature_keys=['serving_default'])

converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.optimizations = []
# converter.target_spec.supported_types = [tf.float32]

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

## TFLite Interpreter to check input shape
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)

# raise NotImplementedError

# # Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# tflite_model = converter.convert()

# # Save the model
# with open(tflite_model_path, 'wb') as f:
#     f.write(tflite_model)