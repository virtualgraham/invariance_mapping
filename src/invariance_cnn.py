import keras
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.utils import Sequence

image_size = 128

def build_cnn():

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(32, 32), input_shape=(264, 264, 1)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(32, 32)))
    model.add(BatchNormalization())

    model.add(Conv2D(1, kernel_size=(3, 3)))
    model.add(BatchNormalization())

    return model

def open_and_prepare_image(image_path):
    image = cv2.imread(image_path)
    scale_dims = get_scaled_dims((image.shape[0], image.shape[1]), image_size)
    image = cv2.resize(image, scale_dims)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = crop_to_square(image)
    image = np.expand_dims(image, axis=2)
    return image

def open_and_prepare_heat_map(heat_map_path):
    image = cv2.imread(heat_map_path)
    return image

def get_heat_map_path(image_path, heatmap_directory):
    image_path = path.splitext(image_path)[0]+'.png'
    name = path.basename(image_path)
    directory = path.join(heatmap_directory, path.basename(path.dirname(image_path)))
    return path.join(directory, name)

class DataSequence(Sequence):

    def __init__(self, hpatches_sequences_directory, heatmap_directory, batch_size, mode='train'):
        
        self.batch_size = batch_size
        self.mode = mode
        self.paths = []
        
        hpatch_sequence_directories = [path.join(hpatches_sequences_directory, d) for d in listdir(hpatches_sequences_directory) if path.isdir(path.join(hpatches_sequences_directory, d))]

        for d in hpatch_sequence_directories:
            for f in listdir(d) if f.endswith('.ppm'):
                image_path = path.join(d, f)
                heat_map_path = get_heat_map_path(image_path)
                self.paths.append((image_path, heat_map_path))

    def __len__(self):
        # compute number of batches to yield
        return int(math.ceil(len(self.paths) / float(self.batch_size)))

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.paths))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def __getitem__(self, idx):

        batch_paths = self.paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        images = []
        heat_maps = []

        for image_path, heat_map_path in batch_paths:
            images.append(open_and_prepare_image(image_path))
            heat_maps.append(open_and_prepare_heat_map(heat_map_path))

        return images, heat_maps

cnn = build_cnn()
print(cnn.summary())