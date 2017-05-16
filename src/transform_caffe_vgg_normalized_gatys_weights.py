import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.applications.vgg19 import VGG19

caffe.set_mode_cpu()

net = caffe.Net('../models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt', '../models/vgg/vgg_normalised.caffemodel', caffe.TEST)

model = VGG19(input_tensor=None, weights="imagenet", include_top=False, pooling="avg")
#model.load_weights("../models/normalized.h5")
model_layers = dict([(layer.name, layer.output) for layer in model.layers])
normlayernames = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
deflayerids =  [1, 2, 4, 5, 7, 8, 9, 10, 12,13,14,15, 17, 18, 19, 20]
for normlayername, deflayerid in zip(normlayernames, deflayerids):
    #Filter weights
    #net.params[normlayername][0].data.transpose((2, 3, 1, 0))/model.layers[deflayerid].get_weights()[0]
    layerweights = model.layers[deflayerid].get_weights()
    layerweights[0] = net.params[normlayername][0].data.transpose((2, 3, 1, 0))
    layerweights[1] = net.params[normlayername][1].data
    model.layers[deflayerid].set_weights(layerweights)
model.save_weights("../models/gatysnormalized.h5")

