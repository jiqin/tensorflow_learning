import os, sys
import argparse
import numpy as np
import time
import cv2

GESTURE_TYPE = ['rock', 'paper', 'scissor']
TRAIN_RATE = 0.8
IMAGE_SIZE = (64, 64)

def reshape_images(images):
    images = np.reshape(images, [-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    return images
    
    
def reshape_labels(labels):
    labels = np.reshape(labels, [-1, 3])
    labels = labels.astype(np.float32)
    return labels
    
    
def load_data(root_path, max_image_num):
    # Data loading and preprocessing
    
    Y_Base = [[1,0,0], [0,1,0], [0,0,1]]
    
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    jpg_files = []
    for i, gesture_type in enumerate(GESTURE_TYPE):
        folder = os.path.join(root_path, gesture_type)
        jpg_files.extend([(os.path.join(folder, f), i) for f in os.listdir(folder) if f.endswith('.jpg')])
    
    np.random.shuffle(jpg_files)
    jpg_files = jpg_files[0:max_image_num]
    train_num = int(len(jpg_files) * TRAIN_RATE)
    
    for file in jpg_files[0:train_num]:
        train_images.append(cv2.resize(cv2.imread(file[0]), IMAGE_SIZE))
        train_labels.append(Y_Base[file[1]])
        
    for file in jpg_files[train_num:]:
        test_images.append(cv2.resize(cv2.imread(file[0]), IMAGE_SIZE))
        test_labels.append(Y_Base[file[1]])
                
    train_images = reshape_images(train_images)
    train_labels = reshape_labels(train_labels)
    test_images = reshape_images(test_images)
    test_labels = reshape_labels(test_labels)
    
    return train_images, train_labels, test_images, test_labels


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
    file_names = [f for f in os.listdir(model_parameter_path) if f.endswith('.index')]
    indexes = list(map(lambda f: int(f.split('.')[0].split('_')[1]), file_names))
    return max(indexes) if len(indexes) > 0 else None
        

def get_model_parameter_path_name(model_parameter_path, index):
    return os.path.join(model_parameter_path, 'model_{0:05d}'.format(index))
    

def load_model_parameter(model, model_parameter_path, index):
    file = get_model_parameter_path_name(model_parameter_path, index)
    print('load model parameter: ' + file)
    model.load(file)
    
    
def save_model_parameter(model, model_parameter_path, index):
    file = get_model_parameter_path_name(model_parameter_path, index)
    print('save model parameter: ' + file)
    model.save(file)
    
    
def train_model(model, epoch_num, train_x, train_y, test_x, test_y):
    model.fit({'input': train_x}, {'target': train_y}, n_epoch=epoch_num,
                validation_set=({'input': test_x}, {'target': test_y}),
                snapshot_step=100, show_metric=True, run_id='convnet_hand_gesture')
    for i in range(10):
        print(test_y[i], model.predict([test_x[i]])[0])    
    
    
def test_model(model):
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        cv2.imshow("capture", frame)
        ret = model.predict([cv2.resize(frame, IMAGE_SIZE)])
        print(ret, np.argmax(ret), GESTURE_TYPE[np.argmax(ret)])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
def main():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--input_image_root_path', dest='input_image_root_path', default='images_input_rgb_480_640')
    parser.add_argument('--model_parameter_path', dest='model_parameter_path', default='model_parameter')
    parser.add_argument('--max_input_count', dest='max_input_count', type=int, default=1000, help='max input image count for train')
    parser.add_argument('--epoch_num', dest='epoch_num', type=int, default=20, help='epoch_num')
    parser.add_argument('--train_model', dest='train_model', action='store_true', default=False, help='train model or run model')
    args = parser.parse_args()
    
    model = create_model(0.0003)
    
    latest_model_index = get_latest_model_parameter_index(args.model_parameter_path)
    if latest_model_index is not None:
        load_model_parameter(model, args.model_parameter_path, latest_model_index)
    else:
        latest_model_index = 0
        
    if args.train_model:
        train_x, train_y, test_x, test_y = load_data(args.input_image_root_path, args.max_input_count)    
        while True:
            train_model(model, args.epoch_num, train_x, train_y, test_x, test_y)
            latest_model_index += 1
            save_model_parameter(model, args.model_parameter_path, latest_model_index)
            time.sleep(3)
    else:
        test_model(model)
        
    
if __name__ == '__main__':
    main()