from PIL import Image
import math
import numpy as np
import time
from PIL import Image
from keras import backend
from keras.models import Model
from keras.applications.vgg19 import VGG19
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

from ScipyOptimizer import ScipyOptimizer


content_path = "../images/inputs/tubingen.jpg"

# Network related
#content_layer_name = "block4_conv2"
#style_layers_names = ["block1_conv1", "block2_conv1", "block3_conv1",
 #                     "block4_conv1", "block5_conv1"]
#style_layers_weights = [0.2, 0.2, 0.2, 0.2, 0.2]     # Must be of the same size as style_layers_names
meanRGB = [123.68, 116.779, 103.939]
#init_result_image = "STYLE" # CAN BE : NOISE, STYLE, CONTENT
width = 512
height = 512

# Transforms an image object into an array ready to be fed to VGG
def preprocess_image(image):
    image = image.resize((height, width))
    array = np.asarray(image, dtype="float32")
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


##### Images Loading
content_image = Image.open(content_path)

content_array = preprocess_image(content_image)
# Creating placeholders in tensorflow
content_tensor = backend.constant(content_array)
result_tensor = backend.placeholder((1, height, width, 3))

# The tensor that will be fed to the VGG network
# The first dimension is used to access the content, style or result image.
# The remaining dimensions are used to access height, width and color channel, in that order.
input_tensor = backend.concatenate([content_tensor, result_tensor], axis=0)

from keras import backend as K

###### Model Loading
model = VGG19(input_tensor=None, weights="imagenet", include_top=False, pooling="avg")
model_layers = dict([(layer.name, layer.output) for layer in model.layers])
sorted_layers = sorted(model_layers.keys())
conv_layer_names = [layer for layer in sorted_layers if 'conv' in layer]

for layer_name in conv_layer_names:
    output_layer = model.get_layer(layer_name)
    get_layer_output = K.function([model.layers[0].input],
                                  [output_layer.output])
    layer_output = get_layer_output([content_array])[0]
    #layer_output is 1, 512, 512, 64
    activation_per_filter = np.sum(layer_output, axis=1)
    activation_per_filter = np.sum(activation_per_filter, axis=1)

    #each filter outputs 512x512
    activation_count_per_filter = layer_output.shape[1]*layer_output.shape[2]
    filter_scale_factor = activation_count_per_filter / activation_per_filter
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

#final evaluation
output_layer = model.get_layer(layer_name)
get_layer_output = K.function([model.layers[0].input],
                                  [output_layer.output])
layer_output = get_layer_output([content_array])[0]

#layer_output is 1, 512, 512, 64
activation_per_filter = np.sum(layer_output, axis=1)
activation_per_filter = np.sum(activation_per_filter, axis=1)

#each filter outputs 512x512
activation_count_per_filter = layer_output.shape[1]*layer_output.shape[2]
mean_activation = activation_per_filter / activation_count_per_filter
#should be 64
print(np.sum(mean_activation))
