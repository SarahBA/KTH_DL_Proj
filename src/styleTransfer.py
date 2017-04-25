from PIL import Image
import numpy as np

from PIL import Image
from keras import backend
from keras.models import Model
from keras.applications.vgg19 import VGG19

# Parameters
style_path = '../images/elephant.png'
content_path = '../images/picasso.jpg'
height = 512
width = 512

# Images Loading
meanRGB = [103.939, 116.779, 123.68];

content_image = Image.open(content_path)
content_image = content_image.resize((height, width))
style_image = Image.open(style_path)
style_image = style_image.resize((height, width))

content_array = np.asarray(content_image, dtype='float32')
style_array = np.asarray(style_image, dtype='float32')

content_array[:, :, 0] -= meanRGB[0]
content_array[:, :, 1] -= meanRGB[1]
content_array[:, :, 2] -= meanRGB[2]
content_array = content_array[:, :, ::-1] # Reordering from RGB to BGR to fit 

style_array[:, :, 0] -= meanRGB[0]
style_array[:, :, 1] -= meanRGB[1]
style_array[:, :, 2] -= meanRGB[2]
style_array = style_array[:, :, ::-1] # Reordering from RGB to BGR

# Model Loading
model = VGG19(weights='imagenet', include_top=False, pooling='avg')
layers = dict([(layer.name, layer.output) for layer in model.layers])

for layer in layers:
	print(layer)


# Show result image
x = content_array.reshape((height, width, 3)); 

x = x[:, :, ::-1]
x[:, :, 0] += meanRGB[0]
x[:, :, 1] += meanRGB[1]
x[:, :, 2] += meanRGB[2]
x = np.clip(x, 0, 255).astype('uint8')

im = Image.fromarray(x)
im.show()

