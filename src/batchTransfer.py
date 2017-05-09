import os

# Network parameters
content_weights = [1, 5, 1]
style_weights = [100, 100, 100]
content_paths = ["../images/inputs/tubingen.jpg", "../images/inputs/tubingen.jpg"]
style_paths = ["../images/inputs/Femme_nue_assise.jpg", "../images/inputs/Composition_VII.jpg"]
result_paths = ["tubingen_femme", "tubingen_composition"]
regularizations = [0]

# Batch parameters
number_iterations = 200
batch_output_base_path = '../images/run/b'
batch_number = 5

# Running the batches
for regularization in regularizations:
	for content_weight, style_weight, content_path, style_path, result_path_base_name in zip(content_weights, style_weights, content_paths, style_paths, result_paths):
		batch_number += 1
		batch_output_path = batch_output_base_path + str(batch_number)
		os.system('python3 ./styleTransfer.py -c %s -s %s -o %s -sw %d -cw %d -rw %d -mi %d --size 128' % (content_path, style_path, batch_output_path, style_weight, content_weight, regularization, number_iterations))
