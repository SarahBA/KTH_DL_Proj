from PIL import Image
import numpy as np
import time
from PIL import Image
from keras import backend
from keras.models import Model
from keras.applications.vgg19 import VGG19
from scipy.optimize import fmin_l_bfgs_b


###### Function definitions

# Computes the content loss value for the content features and result features (both are tensors)
def content_loss(content_feature, result_feature):
	return (1/2) * backend.sum(backend.square(content_feature - result_feature))


##### Parameters
content_weight = 1
style_weight = 1
style_path = '../images/picasso.jpg'
content_path = '../images/elephant.png'
max_iter = 10
height = 256
width = 256

content_layer_name = 'block2_conv2';
style_layers_names = []


##### Images Loading
meanRGB = [103.939, 116.779, 123.68];

content_image = Image.open(content_path)
content_image = content_image.resize((height, width))
style_image = Image.open(style_path)
style_image = style_image.resize((height, width))

content_array = np.asarray(content_image, dtype='float32')
content_array = np.expand_dims(content_array, axis=0)		# Expanding dimensions in order to concatenate the images together
style_array = np.asarray(style_image, dtype='float32')
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
content_tf = backend.variable(content_array)
style_tf = backend.variable(style_array)
result_tf = backend.placeholder((1, height, width, 3))

# The tensor that will be fed to the VGG19 network
# The first dimension is used to access the content, style or result image. 
# The remaining dimensions are used to access height, width and color channel, in that order.
input_tensor = backend.concatenate([content_tf, style_tf, result_tf], axis=0)


###### Model Loading
model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False, pooling='avg')
model_layers = dict([(layer.name, layer.output) for layer in model.layers])


###### Defining the loss function
loss = backend.variable(0)
loss += content_weight * content_loss(model_layers[content_layer_name][0, :, :, :], model_layers[content_layer_name][2, :, :, :])


###### Generating the result image with scipy's L_BFGS_B optimizer

grads = backend.gradients(loss, result_tf)

# When this function is called, tensorflow will evaluate the output
# Used as an interface with "non-tensorflow" python code
outputs = [loss]
outputs += grads
tf_lossgrad = backend.function([result_tf], outputs)

# This function is the one to be used by the scipy optimizer, 
# provides an interface with scipy
def evaluate_loss_gradient(input_image):
	input_image = input_image.reshape((1, height, width, 3))
	outs = tf_lossgrad([input_image])
	loss_value = outs[0]
	grad_values = outs[1].flatten().astype('float64')
	return loss_value, grad_values

# Object created to store the computations
# Scipy evaluates the gradient and the loss separately which is inefficient, 
# thus the need to memorize the results in an object
class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, input_image):
        loss_value, grad_values = evaluate_loss_gradient(input_image)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, input_image):
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()


result_image = np.random.uniform(0, 255, (1, height, width, 3)) - 128

try:
	for i in range(max_iter):
	    print('Iteration', i)
	    start_time = time.time()
	    result_image, loss, info = fmin_l_bfgs_b(evaluator.loss, result_image.flatten(),
	                                     fprime=evaluator.grads, maxfun=20)
	    print('New loss value:', loss)
	    end_time = time.time()
	    print('Iteration %d completed in %ds' % (i, end_time - start_time))
except KeyboardInterrupt:
	pass

###### Showing result image
# result_image = content_array[0, :, :, :]
result_image = result_image.reshape((height, width, 3))

result_image = result_image[:, :, ::-1]
result_image[:, :, 0] += meanRGB[0]
result_image[:, :, 1] += meanRGB[1]
result_image[:, :, 2] += meanRGB[2]
result_image = np.clip(result_image, 0, 255).astype('uint8')

im = Image.fromarray(result_image)
im.show()