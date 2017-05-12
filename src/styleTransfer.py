from PIL import Image
import math
import numpy as np
import time
from PIL import Image
from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import argparse

from ScipyOptimizer import ScipyOptimizer

##### Arguments definition
parser = argparse.ArgumentParser(description='Apply the style of an image onto another one.',
								 usage = 'styleTransfer.py -s <style_image> -c <content_image>')

parser.add_argument('-s', '--style-img', type=str, required=True, help='image to use as style')
parser.add_argument('-c', '--content-img', type=str, required=True, help='image to use as content')
parser.add_argument('-mi', '--max-iter', type=int, required=False, default=10,
					help='the maximum number of iterations for generating the result image')
parser.add_argument('-sw', '--style-weight', type=float, required=False, default=1000,
					help='weight of the style when generating the image')
parser.add_argument('-cw', '--content-weight', type=float, required=False, default=5,
					help='weight of the content when generating the image')
parser.add_argument('-rw', '--reg-weight', type=float, required=False, default=1,
					help='weight of the noise elimination when generating the image')
parser.add_argument('-i', '--init', type=str, required=False, default='noise',
					help='initialization strategy (can be content, style, or noise)')
parser.add_argument('--size', type=int, required=False, default=512,
					help='size of the result image in pixels (height and width are set to the same value)')
parser.add_argument('-o', '--output', type=str, required=False, default='./result_',
					help='define the base path and name for the generated result images')
parser.add_argument('-cl', '--content-layer', type=str, required=False, default='block4_conv2',
					help='the name of the layer to use for the content function')
parser.add_argument('-sl', '--style-layers', type=str, nargs='+', required=False,
					default=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],
					help='the name of the layers to use for the style function')
parser.add_argument('-m', '--model', type=str, required=False, default='VGG19',
					help='the CNN to use (can be VGG16 or VGG19)')


##### Hard Coded Parameters
meanRGB = [123.68, 116.779, 103.939]

###### Functions definitions

# Computes the content loss value for the content features and result features (both are tensors)
def content_loss(content_feature, result_feature):
    return (1/2) * backend.sum(backend.square(content_feature - result_feature))


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram


def style_loss(style, combination, height, width):
	A = gram_matrix(style)
	G = gram_matrix(combination)
	channels = 3
	size = height * width
	return backend.sum(backend.square(G - A)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x, height, width):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


# Transforms an image object into an array ready to be fed to VGG
def preprocess_image(image, height, width):
    image = image.resize((height, width))
    array = np.asarray(image, dtype="float32")
    array = np.expand_dims(array, axis=0) # Expanding dimensions in order to concatenate the images together
    array[:, :, :, 0] -= meanRGB[0] # Subtracting the mean values
    array[:, :, :, 1] -= meanRGB[1]
    array[:, :, :, 2] -= meanRGB[2]
    array = array[:, :, :, ::-1] # Reordering from RGB to BGR to fit VGG19
    return array


# Transforms an array representing an image into a scipy image object
def deprocess_array(array, height, width):
    deprocessed_array = np.copy(array)
    deprocessed_array = deprocessed_array.reshape((height, width, 3))
    deprocessed_array = deprocessed_array[:, :, ::-1] # BGR to RGB
    deprocessed_array[:, :, 0] += meanRGB[0]
    deprocessed_array[:, :, 1] += meanRGB[1]
    deprocessed_array[:, :, 2] += meanRGB[2]
    deprocessed_array = np.clip(deprocessed_array, 0, 255).astype("uint8")
    image = Image.fromarray(deprocessed_array)
    return image

# Main function
def main(args):
	##### Set parameters from arguments
	content_path = args.content_img
	style_path = args.style_img
	max_iter = args.max_iter
	content_weight = args.content_weight
	style_weight = args.style_weight
	regularization = args.reg_weight

	height = args.size
	width = args.size
	result_image_pathprefix = args.output
	content_layer_name = args.content_layer
	style_layers_names = args.style_layers

	model_name = args.model
	if model_name != 'VGG16' or model_name != 'VGG19':
		model_name = 'VGG19'

	init_result_image = args.init
	if init_result_image != 'style' or init_result_image != 'content' or init_result_image != 'noise':
		init_result_image = 'noise'

	style_layers_weights = [1.0/len(style_layers_names) for i in range(len(style_layers_names))]

	print('\nRunning for maximum %d iterations' % max_iter)
	print('Using %s network' % model_name)
	print('with content_weight = %5.3f    style_weight = %d    regularization = %3.1f' %(content_weight, style_weight, regularization))
	print('Content layer is %s' % content_layer_name)
	print('Style layers are %s' % style_layers_names)
	print('Style layers weights are %s' % style_layers_weights)
	print('Image size of %dx%dpx, initialization strategy is %s \n' % (height, width, init_result_image))


	##### Images Loading
	content_image = Image.open(content_path)
	style_image = Image.open(style_path)

	content_array = preprocess_image(content_image, height, width)
	style_array = preprocess_image(style_image, height, width)

	# Creating placeholders in tensorflow
	content_tensor = backend.constant(content_array)
	style_tensor = backend.constant(style_array)
	result_tensor = backend.placeholder((1, height, width, 3))

	# The tensor that will be fed to the VGG network
	# The first dimension is used to access the content, style or result image.
	# The remaining dimensions are used to access height, width and color channel, in that order.
	input_tensor = backend.concatenate([content_tensor, style_tensor, result_tensor], axis=0)

	###### Model Loading
	if model_name == 'VGG16':
		model = VGG16(input_tensor=input_tensor, weights="imagenet", include_top=False)
	else:
		model = VGG19(input_tensor=input_tensor, weights="imagenet", include_top=False, pooling="avg")

	#
	#   model.load_weights("../models/normalized.h5")
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
	    sl = style_loss(style_features, combination_features, height, width)
	    total_style_loss += style_layers_weights[index] * sl

	loss += style_weight * total_style_loss

	# Regularization of the result image
	loss += regularization * total_variation_loss(result_tensor, height, width)


	###### Generating the result image
	scipyOpt = ScipyOptimizer(loss, result_tensor)

	# Initializing the result image
	if init_result_image == "style":
	    result_array = np.copy(style_array)
	elif init_result_image == "content":
	    result_array = np.copy(content_array)
	else:
	    result_array = np.random.uniform(0, 255, (1, height, width, 3)) - 128

	# Starting iterations
	total_time = 0
	for i in range(max_iter):
		print("======= Starting iteration %d =======" % (i+1))
		start_time = time.time()
		result_array, loss_value, info = scipyOpt.optimize(result_array)
		end_time = time.time()
		ellapsed_time = (end_time - start_time)
		total_time += ellapsed_time
		estimated_time = (total_time/(i+1) * (max_iter-i-1))

		result_image_fullpath = result_image_pathprefix + "_iter_" + str(i+1) + "_r" + str(regularization) + "_s" + str(style_weight) + "_c" + str(content_weight) +".png"
		im = deprocess_array(result_array, height, width)
		print("Saving image for this iteration as %s" % result_image_fullpath)
		imsave(result_image_fullpath, im)

		print("New loss value", loss_value)
		print("Iteration completed in %ds" % ellapsed_time)
		print("Estimated time left : %dm%ds" % (math.floor(estimated_time/60.), estimated_time%60))

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
