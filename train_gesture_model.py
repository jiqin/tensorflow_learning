import argparse
import time
import numpy as np
from utils import load_image_data, create_model, load_latest_model_parameter, save_model_parameter


def train_model(model, epoch_num, train_x, train_y, test_x, test_y):
    model.fit({'input': train_x}, {'target': train_y}, n_epoch=epoch_num,
                validation_set=({'input': test_x}, {'target': test_y}),
                snapshot_step=100, show_metric=True, run_id='convnet_hand_gesture')
    for i in range(5):
        y1 = model.predict([train_x[i]])[0]
        print('train:', np.argmax(train_y[i]), np.argmax(y1), y1[np.argmax(y1)], y1)
    for i in range(5):
        y1 = model.predict([test_x[i]])[0]
        print('test:', np.argmax(test_y[i]), np.argmax(y1), y1[np.argmax(y1)], y1)
        
        

def main():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--image_root', dest='image_root', default='inputs_128_128', help='saved image root')
    parser.add_argument('--model_parameter_path', dest='model_parameter_path', default='model_parameter')
    parser.add_argument('--max_input_count', dest='max_input_count', type=int, default=10000, help='max input image count for train')
    parser.add_argument('--epoch_num', dest='epoch_num', type=int, default=20, help='epoch_num')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0001, help='learning rate')
    
    args = parser.parse_args()
    
    train_x, train_y, test_x, test_y = load_image_data(args.image_root, args.max_input_count) 
    
    model = create_model(args.learning_rate)
    
    latest_model_index = load_latest_model_parameter(model, args.model_parameter_path)
    while True:
        train_model(model, args.epoch_num, train_x, train_y, test_x, test_y)
        latest_model_index += 1
        save_model_parameter(model, args.model_parameter_path, latest_model_index)
        time.sleep(3)
        
    
if __name__ == '__main__':
    main()