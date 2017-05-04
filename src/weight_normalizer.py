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
model_layer_outputs = dict([(layer.name, layer.output) for layer in model.layers])
model_layers = dict([(layer.name, layer.output) for layer in model.layers])

predictions = model.predict(content_array)
###### Defining the total loss function
#loss = backend.variable(0)

# The Content loss
#loss += content_weight * content_loss(model_layers[content_layer_name][0, :, :, :], model_layers[content_layer_name][2, :, :, :])

#loss += style_weight * total_style_loss

#get_0rd_layer_output = K.function([model.layers[0].input],
 #                                 [model.layers[1].output])
get_0rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])


layer_output = get_0rd_layer_output([content_array])[0]
#layer_output is 1, 512, 512, 64
activation_per_filter = np.sum(layer_output, axis=(1, 2))
#each filter outputs 512x512
activation_count_per_filter = layer_output.shape[1]*layer_output.shape[2]
mean_activation = activation_per_filter / activation_count_per_filter
#1x64
filter_scale_factor = 1 / mean_activation
#total_count = layer_output.size
layer_weights = model.layers[1].get_weights()
#3x3x64
filter_weights = layer_weights[0]

filter_weights_rescaled = filter_weights * filter_scale_factor
layer_weights[0] = filter_weights_rescaled
model.layers[1].set_weights(layer_weights)
