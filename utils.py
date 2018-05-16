import os
import numpy as np
import cv2

GESTURE_TYPE = ['rock', 'paper', 'scissor']
LABEL_BASE = [[1,0,0], [0,1,0], [0,0,1]]

IMAGE_SIZE = (128, 128)
TRAIN_RATE = 0.8


def resize_image_to_predefind_size(image):
    return cv2.resize(image, IMAGE_SIZE)
    
    
def save_image(image, image_id, image_root_path, gesture_type):
    folder = os.path.join(image_root_path, gesture_type)
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, 'image_{0:05d}.jpg'.format(image_id)), resize_image_to_predefind_size(image))

    
def load_image(jpg_file):
    return cv2.imread(jpg_file)
    

def load_image_data(image_root_path, max_image_num):    
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    jpg_file_and_label_index_pairs = []
    for label_index, gesture_type in enumerate(GESTURE_TYPE):
        folder = os.path.join(image_root_path, gesture_type)
        jpg_file_and_label_index_pairs.extend([(os.path.join(folder, f), label_index) for f in os.listdir(folder) if f.endswith('.jpg')])
    
    np.random.shuffle(jpg_file_and_label_index_pairs)
    jpg_file_and_label_index_pairs = jpg_file_and_label_index_pairs[0:max_image_num]
    train_num = int(len(jpg_file_and_label_index_pairs) * TRAIN_RATE)
    
    for jpg_file_and_label_index_pair in jpg_file_and_label_index_pairs[0:train_num]:
        train_images.append(load_image(jpg_file_and_label_index_pair[0]))
        train_labels.append(LABEL_BASE[jpg_file_and_label_index_pair[1]])
        
    for jpg_file_and_label_index_pair in jpg_file_and_label_index_pairs[train_num:]:
        test_images.append(load_image(jpg_file_and_label_index_pair[0]))
        test_labels.append(LABEL_BASE[jpg_file_and_label_index_pair[1]])
                
    train_images = normalize_images(train_images)
    train_labels = normalize_label(train_labels)
    test_images = normalize_images(test_images)
    test_labels = normalize_label(test_labels)
    
    return train_images, train_labels, test_images, test_labels
    
    
def normalize_images(images):
    images = list(map(resize_image_to_predefind_size, images))
    images = np.reshape(images, [-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    return images
    
    
def normalize_label(labels):
    labels = np.reshape(labels, [-1, 3])
    labels = labels.astype(np.float32)
    return labels
    
    
def create_model(learning_rate):
    import tflearn
    import tensorflow as tf
    from tflearn.layers.core import dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.normalization import local_response_normalization
    from tflearn.layers.estimator import regression
    
    network = tflearn.input_data(shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=learning_rate,
                            loss='categorical_crossentropy', name='target')
    
    model = tflearn.DNN(network, tensorboard_verbose=0)
    return model
    
    
def get_latest_model_parameter_index(model_parameter_path):
    os.makedirs(model_parameter_path, exist_ok=True)
    file_names = [f for f in os.listdir(model_parameter_path) if f.endswith('.index')]
    indexes = list(map(lambda f: int(f.split('.')[0].split('_')[1]), file_names))
    return max(indexes) if len(indexes) > 0 else None
        

def get_model_parameter_path_name(model_parameter_path, index):
    return os.path.join(model_parameter_path, 'model_{0:05d}'.format(index))
    

def load_latest_model_parameter(model, model_parameter_path):
    latest_model_index = get_latest_model_parameter_index(model_parameter_path)
    if latest_model_index is not None:
        file = get_model_parameter_path_name(model_parameter_path, latest_model_index)
        print('load model parameter: ' + file)
        model.load(file)
    else:
        latest_model_index = 0
    return latest_model_index
    
    
def save_model_parameter(model, model_parameter_path, index):
    file = get_model_parameter_path_name(model_parameter_path, index)
    print('save model parameter: ' + file)
    model.save(file)