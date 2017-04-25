from PIL import Image
import numpy as np

from PIL import Image
from keras import backend
from keras.models import Model
from keras.applications.vgg19 import VGG19

##### Parameters
style_path = '../images/elephant.png'
content_path = '../images/picasso.jpg'
height = 512
width = 512

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

# Create placeholders in tensorflow
content_image = backend.variable(content_array)
style_image = backend.variable(style_array)
result_image = backend.placeholder((1, height, width, 3))

# The tensor that will be fed to the VGG19 network
# The first dimension is used to access the content, style or result image. 
# The remaining dimensions are used to access height, width and color channel, in that order.
input_tensor = backend.concatenate([content_image, style_image, result_image], axis=0)


###### Model Loading
model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False, pooling='avg')
layers = dict([(layer.name, layer.output) for layer in model.layers])



###### Show result image
x = content_array[0, :, :, :]
x = x.reshape((height, width, 3)); 

x = x[:, :, ::-1]
x[:, :, 0] += meanRGB[0]
x[:, :, 1] += meanRGB[1]
x[:, :, 2] += meanRGB[2]
x = np.clip(x, 0, 255).astype('uint8')

im = Image.fromarray(x)
im.show()



###### Function definitions

# Computes the content loss value for the content features and result features (both are tensors)
def contentLoss(content_feature, result_feature)
	return (1/2) * backend.sum(backend.square(backend.sub(content_feature, result_feature)))

