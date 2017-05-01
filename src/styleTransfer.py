from PIL import Image
import math
import numpy as np
import time
from PIL import Image
from keras import backend
from keras.models import Model
from keras.applications.vgg19 import VGG19
from scipy.optimize import fmin_l_bfgs_b

from ScipyOptimizer import ScipyOptimizer


##### Parameters
content_weight = 1 # 0.05 when 512*512
style_weight = 10 # 5.0 when 512*512
regularization = 10 # 1.0 when 512*512 (0 in the original paper) 
style_path = "../images/scream.jpg"
content_path = "../images/tubingen.jpg"
max_iter = 10
height = 128    # Size of images in the paper : 512*512
width = 128

content_layer_name = "block4_conv2"
style_layers_names = ["block1_conv1", "block2_conv1", "block3_conv1",
                      "block4_conv1", "block5_conv1"]
style_layers_weights = [0.2, 0.2, 0.2, 0.2, 0.2]     # Must be of the same size as style_layers_names


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


##### Images Loading
meanRGB = [123.68, 116.779, 103.939];

content_image = Image.open(content_path)
content_image = content_image.resize((height, width))
style_image = Image.open(style_path)
style_image = style_image.resize((height, width))

content_array = np.asarray(content_image, dtype="float32")
content_array = np.expand_dims(content_array, axis=0)		# Expanding dimensions in order to concatenate the images together
style_array = np.asarray(style_image, dtype="float32")
style_array =np.expand_dims(style_array, axis=0)

# Transform the input to correspond to what VGG19 wants
content_array[:, :, :, 0] -= meanRGB[0]
content_array[:, :, :, 1] -= meanRGB[1]
content_array[:, :, :, 2] -= meanRGB[2]
content_array = content_array[:, :, :, ::-1] # Reordering from RGB to BGR to fit VGG19

style_array[:, :, :, 0] -= meanRGB[0]
style_array[:, :, :, 1] -= meanRGB[1]
style_array[:, :, :, 2] -= meanRGB[2]
style_array = style_array[:, :, :, ::-1]

# Creating placeholders in tensorflow
content_tensor = backend.constant(content_array)
style_tensor = backend.constant(style_array)
result_tensor = backend.placeholder((1, height, width, 3))

# The tensor that will be fed to the VGG19 network
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
result_image = np.random.uniform(0, 255, (1, height, width, 3)) - 128

total_time = 0
try:
    for i in range(max_iter):
        print("======= Starting iteration %d =======" % (i+1))
        start_time = time.time()
        result_image, loss_value, info = scipyOpt.optimize(result_image)
        end_time = time.time()
        ellapsed_time = (end_time - start_time)
        total_time += ellapsed_time
        estimated_time = (total_time/(i+1) * (max_iter-i-1))
        print("New loss value", loss_value)
        print("Iteration completed in %ds" % ellapsed_time)
        print("Estimated time left : %dm%ds" % (math.floor(estimated_time/60.), estimated_time%60))
except KeyboardInterrupt:
    pass


###### Showing result image
result_image = result_image.reshape((height, width, 3))

result_image = result_image[:, :, ::-1]
result_image[:, :, 0] += meanRGB[0]
result_image[:, :, 1] += meanRGB[1]
result_image[:, :, 2] += meanRGB[2]
result_image = np.clip(result_image, 0, 255).astype("uint8")

im = Image.fromarray(result_image)
im.show()
