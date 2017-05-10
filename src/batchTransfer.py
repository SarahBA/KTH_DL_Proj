import os

# Network parameters
# ALL THE FOLLOWING ARRAYS MUST BE OF SAME LENGTH
# The batch script will simply run the style transfer algorithm will the parameters
# extracted from the first indexes of the arrays, then the second, etc...

content_weights = [0.025, 0.025]
style_weights = [5, 5]
content_paths = ["../images/inputs/hugo.jpg", "../images/inputs/hugo.jpg"]
style_paths = ["../images/inputs/waves.png", "../images/inputs/chaos.jpg"]
result_prefixes = ["hugo_waves", "hugo_chaos"]
regularizations = [0.1, 0.1]

# Batch parameters
number_iterations = 200
batch_output_base_path = '../images/run/b'
batch_number = 5

# Running the batches
for content_weight, style_weight, content_path, style_path, result_prefix, regularization in zip(content_weights, style_weights, content_paths, style_paths, result_prefixes, regularizations):
		batch_number += 1
		batch_output_path = batch_output_base_path + str(batch_number) + '_' + result_prefix
		os.system('python3 ./styleTransfer.py -c %s -s %s -o %s -sw %d -cw %5.3f -rw %3.1f -mi %d --model %s -cl %s -sl %s' % (content_path, style_path, batch_output_path, style_weight, content_weight, regularization, number_iterations, "VGG16", "block2_conv2", "'block1_conv2' 'block2_conv2' 'block3_conv3' 'block4_conv3' 'block5_conv3'"))
