from PIL import Image
import math
import numpy as np
import time
import glob
from keras import backend as K
from keras.models import Model
from keras.applications.vgg19 import VGG19
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

from ScipyOptimizer import ScipyOptimizer

content_paths = [filename.replace('\\', '/') for filename in glob.glob('d:/DeepLearning/Data/ILSVRC2012_img_val/*.JPEG')] 

# Network related
meanRGB = [123.68, 116.779, 103.939]
width = 512
height = 512

# Transforms an image object into an array ready to be fed to VGG
def preprocess_image(image):
    image = image.resize((height, width))
    array = np.asarray(image, dtype="float32")
    if (len(array.shape) != 3):
        new_array = np.expand_dims(array, axis=3)
        array = np.tile(new_array, 3)
    array = np.expand_dims(array, axis=0) # Expanding dimensions in order to concatenate the images together
    
    array[:, :, :, 0] -= meanRGB[0] # Subtracting the mean values
    array[:, :, :, 1] -= meanRGB[1]
    array[:, :, :, 2] -= meanRGB[2]
    array = array[:, :, :, ::-1] # Reordering from RGB to BGR to fit VGG19
    return array


# Transforms an array representing an image into a scipy image object
def deprocess_array(array):
    deprocessed_array = np.copy(array)
    deprocessed_array = deprocessed_array.reshape((height, width, 3))
    deprocessed_array = deprocessed_array[:, :, ::-1] # BGR to RGB
    deprocessed_array[:, :, 0] += meanRGB[0]
    deprocessed_array[:, :, 1] += meanRGB[1]
    deprocessed_array[:, :, 2] += meanRGB[2]
    deprocessed_array = np.clip(deprocessed_array, 0, 255).astype("uint8")
    image = Image.fromarray(deprocessed_array)
    return image

def load_content_array(content_path):
    content_image = Image.open(content_path)
    content_array = preprocess_image(content_image)
    return content_array

content_arrays = [load_content_array(content_path) for content_path in content_paths]

###### Model Loading
model = VGG19(input_tensor=None, weights="imagenet", include_top=False, pooling="avg")

model_layers = dict([(layer.name, layer.output) for layer in model.layers])
conv_layer_names = [layer for layer in sorted(model_layers.keys()) if 'conv' in layer]
content_count = len(content_arrays)

for layer_name in conv_layer_names:
    output_layer = model.get_layer(layer_name)
    get_layer_output = K.function([model.layers[0].input],
                                      [output_layer.output])
    mean_total = 0
    for content_array in content_arrays:
        layer_output = get_layer_output([content_array])[0]
        activation_count_per_filter = layer_output.shape[1]*layer_output.shape[2]

        #layer_output is 1, 512, 512, 64
        activation_per_filter = np.sum(layer_output, axis=1)
        activation_per_filter = np.sum(activation_per_filter, axis=1)

        mean_contribution = 1/content_count * activation_per_filter / activation_count_per_filter
        mean_total = mean_total + mean_contribution

        #each filter outputs 512x512
    filter_scale_factor = 1 / mean_total
    filter_scale_factor = filter_scale_factor[0]
    layer_weights = output_layer.get_weights()
    #3x3x64
    filter_weights = layer_weights[0]
    filter_bias = layer_weights[1]

    filter_weights_rescaled = filter_scale_factor * filter_weights
    bias_weights_rescaled = filter_scale_factor * filter_bias

    layer_weights[0] = filter_weights_rescaled
    layer_weights[1] = bias_weights_rescaled
    output_layer.set_weights(layer_weights)
    print("Layer completed: " + layer_name)

#save
model_json = model.to_json()
with open("../models/normalized.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../models/normalized.h5")
print("Saved model to disk")

#final evaluation
output_layer = model.get_layer(layer_name)
get_layer_output = K.function([model.layers[0].input],
                                  [output_layer.output])
layer_output = get_layer_output([content_arrays[0]])[0]

#layer_output is 1, 512, 512, 64
activation_per_filter = np.sum(layer_output, axis=1)
activation_per_filter = np.sum(activation_per_filter, axis=1)

#each filter outputs 512x512
activation_count_per_filter = layer_output.shape[1]*layer_output.shape[2]
mean_activation = activation_per_filter / activation_count_per_filter
#should be 64
print(np.sum(mean_activation))

output_layer = model.get_layer(layer_name)
get_layer_output = K.function([model.layers[0].input],
                                  [output_layer.output])
layer_output = get_layer_output([content_arrays[1]])[0]

#layer_output is 1, 512, 512, 64
activation_per_filter = np.sum(layer_output, axis=1)
activation_per_filter = np.sum(activation_per_filter, axis=1)

#each filter outputs 512x512
activation_count_per_filter = layer_output.shape[1]*layer_output.shape[2]
mean_activation = activation_per_filter / activation_count_per_filter
#should be 64
print(np.sum(mean_activation))
