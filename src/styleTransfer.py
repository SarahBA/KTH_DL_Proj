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


##### Parameters
# User defined
content_weight = 5
style_weight = 1000
regularization = 0.1

max_iter = 200
height = 512    # Size of images in the paper : 512*512
width = 512

style_path = "../images/inputs/The_Scream.jpg"
content_path = "../images/inputs/tubingen.jpg"
result_image_pathprefix = "../images/run/b2/tubingen_scream"

# Network related
content_layer_name = "block4_conv2"
style_layers_names = ["block1_conv1", "block2_conv1", "block3_conv1",
                      "block4_conv1", "block5_conv1"]
style_layers_weights = [0.2, 0.2, 0.2, 0.2, 0.2]     # Must be of the same size as style_layers_names
meanRGB = [123.68, 116.779, 103.939]
init_result_image = "NOISE" # CAN BE : NOISE, STYLE, CONTENT

###### Functions definitions

# Computes the content loss value for the content features and result features (both are tensors)
def content_loss(content_feature, result_feature):
    return (1/2) * backend.sum(backend.square(content_feature - result_feature))


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram


def style_loss(style, combination):
    A = gram_matrix(style)
    G = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(G - A)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


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
style_image = Image.open(style_path)

content_array = preprocess_image(content_image)
style_array = preprocess_image(style_image)

# Creating placeholders in tensorflow
content_tensor = backend.constant(content_array)
style_tensor = backend.constant(style_array)
result_tensor = backend.placeholder((1, height, width, 3))

# The tensor that will be fed to the VGG network
# The first dimension is used to access the content, style or result image.
# The remaining dimensions are used to access height, width and color channel, in that order.
input_tensor = backend.concatenate([content_tensor, style_tensor, result_tensor], axis=0)


###### Model Loading
model = VGG19(input_tensor=input_tensor, weights="imagenet", include_top=False, pooling="avg")
model_layers = dict([(layer.name, layer.output) for layer in model.layers])


###### Defining the total loss function
loss = backend.variable(0)

# The Content loss
loss += content_weight * content_loss(model_layers[content_layer_name][0, :, :, :], model_layers[content_layer_name][2, :, :, :])

# The Style loss
total_style_loss = backend.variable(0)
for index, layer_name in enumerate(style_layers_names):
    layer_features = model_layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    total_style_loss += style_layers_weights[index] * sl

loss += style_weight * total_style_loss

# Regularization of the result image
loss += regularization * total_variation_loss(result_tensor)


###### Generating the result image
scipyOpt = ScipyOptimizer(loss, result_tensor)

# Initializing the result image
if init_result_image == "STYLE":
    result_array = np.copy(style_array)
elif init_result_image == "CONTENT":
    result_array = np.copy(content_array)
else:
    result_array = np.random.uniform(0, 255, (1, height, width, 3)) - 128

# Starting iterations
total_time = 0
try:
    for i in range(max_iter):
        print("======= Starting iteration %d =======" % (i+1))
        start_time = time.time()
        result_array, loss_value, info = scipyOpt.optimize(result_array)
        end_time = time.time()
        ellapsed_time = (end_time - start_time)
        total_time += ellapsed_time
        estimated_time = (total_time/(i+1) * (max_iter-i-1))

        result_image_fullpath = result_image_pathprefix + "_iter_" + str(i+1) + ".png"
        im = deprocess_array(result_array)
        print("Saving image for this iteration as %s" % result_image_fullpath)
        imsave(result_image_fullpath, im)

        print("New loss value", loss_value)
        print("Iteration completed in %ds" % ellapsed_time)
        print("Estimated time left : %dm%ds" % (math.floor(estimated_time/60.), estimated_time%60))
except KeyboardInterrupt:
    pass


###### Showing result image
im = deprocess_array(result_array)
im.show()
