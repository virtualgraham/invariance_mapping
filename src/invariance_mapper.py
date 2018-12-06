# for each group of images representing the same scene or object
    # for each image A in group
    #   get L2 descriptor for every 64x64 window in A and place into grid
    #   for each other image B in group
    #       get L2 descriptor for every 64x64 window in B and place into lookup table
    #       map every descriptor in G to the distance to its nearest neighbor in B
    #   this should result in a set of heat maps, one for each other picture
    #   add the heat maps together point for point to create a single heatmap
# every image should have a heat map that represents areas of invariance 

import os
from os import listdir, path
import ntpath

import numpy as np
import cv2
import hnswlib
from L2_Net import L2Net
import math
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

image_size = 640
window_size = 64
descriptor_dimensions = 256

hpatches_sequences_directory = "/home/virtualgraham/hpatches-sequences-release"
heatmap_directory = "/home/virtualgraham/heatmaps"

# list all the sub-directories in the hpatches directory
hpatch_sequence_directories = [path.join(hpatches_sequences_directory, d) for d in listdir(hpatches_sequences_directory) if path.isdir(path.join(hpatches_sequences_directory, d))]
# list all .ppm files for each sub-directory
hpatch_sequences = [[path.join(d, f) for f in listdir(d) if f.endswith('.ppm')] for d in hpatch_sequence_directories]

l2_net = L2Net("L2Net-HP+", True)

def get_scaled_dims(orig_dims, new_smallest_dim):
    if orig_dims[0] < orig_dims[1]:
        new_larger_dim = (new_smallest_dim/orig_dims[0]) * orig_dims[1]
        return (int(new_smallest_dim), int(new_larger_dim))
    else:
        new_larger_dim = (new_smallest_dim/orig_dims[1]) * orig_dims[0]
        return (int(new_larger_dim), int(new_smallest_dim))

def crop_to_square(image):
    diff = abs(image.shape[1] - image.shape[0])
    diff = diff//2
    if image.shape[0] < image.shape[1]:
        return image[:,diff:(diff+image.shape[0])]
    elif image.shape[1] < image.shape[0]:
        return image[diff:(diff+image.shape[1]),:]
    else:
        return image

def open_and_prepare_image(image_path):
    image = cv2.imread(image_path)
    scale_dims = get_scaled_dims((image.shape[0], image.shape[1]), image_size)
    image = cv2.resize(image, scale_dims)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = crop_to_square(image)
    image = np.expand_dims(image, axis=2)
    return image

def save_heat_map(heat_map, image_path):
    print(heat_map.shape)

    image_path = path.splitext(image_path)[0]+'.png'
    name = path.basename(image_path)
    directory = path.join(heatmap_directory, path.basename(path.dirname(image_path)))

    if not path.exists(directory):
        os.makedirs(directory)

    cv2.imwrite(path.join(directory, name), heat_map)

    return

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0] - windowSize[1], stepSize):
		for x in range(0, image.shape[1] - windowSize[0], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def calc_descriptors(image):
    batch = []
    batch_coords = []
    batch_size = 10000

    descriptor_index = hnswlib.Index(space = 'l2', dim = descriptor_dimensions) 
    descriptor_index.init_index(max_elements = (image_size - window_size)**2, ef_construction = 200, M = 16)
    descriptor_index.set_ef(50)

    descriptor_grid = np.empty((image_size - window_size, image_size - window_size, descriptor_dimensions))

    completed_batches = 0

    def run_batch():
        nonlocal completed_batches
        print('batch', completed_batches)
        np_batch = np.array(batch)
        descriptors = l2_net.calc_descriptors(np_batch)
        descriptor_index.add_items(descriptors, [c[0] * (image_size - window_size) + c[1] for c in batch_coords])

        for i in range(0, len(batch_coords)):
            batch_coord = batch_coords[i]
            descriptor = descriptors[i]
            descriptor_grid[batch_coord] = descriptor

        completed_batches += 1
        
    for (x, y, window) in sliding_window(image, stepSize=1, windowSize=(window_size, window_size)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != window_size or window.shape[1] != window_size:
            continue

        batch.append(window)
        batch_coords.append((x, y))

        if len(batch) == batch_size:
            run_batch()
            batch = []
            batch_coords = []

    if len(batch) > 0:
        run_batch()

    return descriptor_index, descriptor_grid

# min_dist = -1
# max_dist = -1

# dist_hist = np.zeros((30,))

for hpatch_sequence in hpatch_sequences:

# for j in range(0, 1):
#     hpatch_sequence = hpatch_sequences[j]

    descriptor_index_dict = {}
    descriptor_grid_dict = {}
    
    print('building indexes')

    for image_path in hpatch_sequence:
        print(image_path)
        image = open_and_prepare_image(image_path)
      
        # print(image.shape)
        # show_img = np.reshape(image[32:96, 32:96, :], (64, 64))
        # print(show_img.shape)
        # imgplot = plt.imshow(show_img)
        # plt.show()

        descriptor_index, descriptor_grid = calc_descriptors(image)
        descriptor_index_dict[image_path] = descriptor_index
        descriptor_grid_dict[image_path] = descriptor_grid

    print('generating heat maps')

    for image_path_a in hpatch_sequence:
        
        heat_maps = []

        for image_path_b in hpatch_sequence:
        
            if image_path_a == image_path_b:
                continue
        
            print(image_path_a, image_path_b)

            descriptor_grid = descriptor_grid_dict[image_path_a]
            descriptor_index = descriptor_index_dict[image_path_b]

            heat_map = np.empty((image_size - window_size, image_size - window_size))

            for x in range(0, image_size - window_size):
                for y in range(0, image_size - window_size): 
                    source = descriptor_grid[x][y]
                    labels, distances = descriptor_index.knn_query(np.array([source]), k=1)
                    d = distances[0, 0]
                    
                    s = d/3.0
                    s = s if s < 1.0 else 1.0
                    heat_map[x, y] = s

                    # if min_dist == -1 or min_dist > d:
                    #     min_dist = d
                    # if max_dist == -1 or max_dist < d:
                    #     max_dist = d

                    # r = int(math.floor(d  * 10))
                    # r = r if r <= 29 else 29
                    # dist_hist[r] = dist_hist[r] + 1

            heat_maps.append(heat_map)

        # combine heat_maps and save as image

        combined_heat_map = np.ones((image_size - window_size, image_size - window_size))

        for x in range(0, image_size - window_size):
            for y in range(0, image_size - window_size): 
                for z in range(0, len(heat_maps)):
                     combined_heat_map[x][y] += (255 * (heat_maps[z][x][y]/len(heat_maps)))

        # imgplot = plt.imshow(combined_heat_map)
        # plt.show()

        save_heat_map(combined_heat_map, image_path_a)

# print('min_dist, max_dist', min_dist, max_dist)
# print(np.array2string(dist_hist))

        

        


        



