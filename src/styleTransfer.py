from PIL import Image
import math
import numpy as np
import time
from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
from vgg19_loader import VGG19
#from keras.applications.vgg19 import VGG19
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import argparse
import scipy
import scipy.misc as spm
import scipy.ndimage as spi
import scipy.sparse as sps
import tensorflow as tf
from ScipyOptimizer import ScipyOptimizer

##### Arguments definition
parser = argparse.ArgumentParser(description='Apply the style of an image onto another one.',
								 usage = 'styleTransfer.py -s <style_image> -c <content_image>')

parser.add_argument('-s', '--style-img', type=str, required=False, help='image to use as style', default="../images/inputs/Der_Schrei.jpg")
parser.add_argument('-c', '--content-img', type=str, required=False, help='image to use as content', default="../images/inputs/tubingen.jpg")
parser.add_argument('-mi', '--max-iter', type=int, required=False, default=200,
					help='the maximum number of iterations for generating the result image')
parser.add_argument('-sw', '--style-weight', type=float, required=False, default=1000,
					help='weight of the style when generating the image')
parser.add_argument('-cw', '--content-weight', type=float, required=False, default=5,
					help='weight of the content when generating the image')
parser.add_argument('-rw', '--reg-weight', type=float, required=False, default=0,
					help='weight of the noise elimination when generating the image')
parser.add_argument('-i', '--init', type=str, required=False, default='noise',
					help='initialization strategy (can be content, style, or noise)')
parser.add_argument('--size', type=int, required=False, default=512,
					help='size of the result image in pixels (height and width are set to the same value)')
parser.add_argument('-o', '--output', type=str, required=False, default='../images/run/b13/result_photo_',
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
use_photo_loss = False
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


def rolling_block(A, block=(3, 3)):
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)

def getlaplacian(i_arr: np.ndarray, consts: np.ndarray, epsilon: float = 0.0000001, win_size: int = 1):
    neb_size = (win_size * 2 + 1) ** 2
    h, w, c = i_arr.shape
    img_size = w * h
    consts = spi.morphology.grey_erosion(consts, footprint=np.ones(shape=(win_size * 2 + 1, win_size * 2 + 1)))
    indsM = np.reshape(np.array(range(img_size)), newshape=(h, w), order='F')
    tlen = int((-consts[win_size:-win_size, win_size:-win_size] + 1).sum() * (neb_size ** 2))
    row_inds = np.zeros(tlen)
    col_inds = np.zeros(tlen)
    vals = np.zeros(tlen)
    l = 0
    for j in range(win_size, w - win_size):
        for i in range(win_size, h - win_size):
            if consts[i, j]:
                continue
            win_inds = indsM[i - win_size:i + win_size + 1, j - win_size: j + win_size + 1]
            win_inds = win_inds.ravel(order='F')
            win_i = i_arr[i - win_size:i + win_size + 1, j - win_size: j + win_size + 1, :]
            win_i = win_i.reshape((neb_size, c), order='F')
            win_mu = np.mean(win_i, axis=0).reshape(1, win_size * 2 + 1)
            win_var = np.linalg.inv(np.matmul(win_i.T, win_i) / neb_size - np.matmul(win_mu.T, win_mu) + epsilon / neb_size * np.identity(c))
            win_i2 = win_i - win_mu
            tvals = (1 + np.matmul(np.matmul(win_i2, win_var), win_i2.T)) / neb_size
            ind_mat = np.broadcast_to(win_inds, (neb_size, neb_size))
            row_inds[l: (neb_size ** 2 + l)] = ind_mat.ravel(order='C')
            col_inds[l: neb_size ** 2 + l] = ind_mat.ravel(order='F')
            vals[l: neb_size ** 2 + l] = tvals.ravel(order='F')
            l += neb_size ** 2
    vals = vals.ravel(order='F')
    row_inds = row_inds.ravel(order='F')
    col_inds = col_inds.ravel(order='F')
    a_sparse = sps.csr_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size), dtype="float32")
    sum_a = a_sparse.sum(axis=1).T.tolist()[0]
    a_sparse = sps.diags([sum_a], [0], shape=(img_size, img_size), dtype="float32") - a_sparse
    return a_sparse

def total_photo_loss(result_tensor, height, width, laplaciantensor):	
	content_array_mult = backend.reshape(result_tensor[:, :, :, 0], (width * height, 1))
	tftensor2 = content_array_mult
#	tftensor2 = backend.constant(content_array_mult)
	multiplied = tf.sparse_tensor_dense_matmul(laplaciantensor, tftensor2)
	content_array_mult_2 = backend.transpose(tftensor2)
	multiplied2 = backend.dot(content_array_mult_2, multiplied)
	return backend.sum(multiplied2)


# Transforms an image object into an array ready to be fed to VGG
def preprocess_image(image, height, width):
    image = image.resize((height, width))
    array = np.asarray(image, dtype="float32")
    array = np.expand_dims(array, axis=0) # Expanding dimensions in order to concatenate the images together
    #array[:, :, :, 0] -= meanRGB[0] # Subtracting the mean values
    #array[:, :, :, 1] -= meanRGB[1]
    #array[:, :, :, 2] -= meanRGB[2]
    array = array[:, :, :, ::-1] # Reordering from RGB to BGR to fit VGG19
    return array


# Transforms an array representing an image into a scipy image object
def deprocess_array(array, height, width):
    deprocessed_array = np.copy(array)
    deprocessed_array = deprocessed_array.reshape((height, width, 3))
    deprocessed_array = deprocessed_array[:, :, ::-1] # BGR to RGB
    #deprocessed_array[:, :, 0] += meanRGB[0]
    #deprocessed_array[:, :, 1] += meanRGB[1]
    #deprocessed_array[:, :, 2] += meanRGB[2]
    deprocessed_array = np.clip(deprocessed_array, 0, 255).astype("uint8")
    image = Image.fromarray(deprocessed_array)
    return image

def img_to_double(image):
    return image / 255.0

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

# Main function
def main(args):
	##### Set parameters from arguments
	content_path = args.content_img
	style_path = args.style_img
	max_iter = args.max_iter
	content_weight = args.content_weight
	style_weight = args.style_weight
	regularization = args.reg_weight
	photo_weight = 10000
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
		model.load_weights("../models/normalized.h5")

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
	#Add photo loss regularization term
	if use_photo_loss:
		content_array_l = img_to_double(content_array)
		content_array_l = content_array[0, :, :, :]
		laplacian = getlaplacian(content_array_l, np.zeros(shape=(height, width)), 1e-7, 1)
		laplaciantensor = convert_sparse_matrix_to_sparse_tensor(laplacian)

		loss += photo_weight * total_photo_loss(result_tensor, height, width, laplaciantensor)

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
