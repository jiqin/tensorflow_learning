import argparse
import sys
import numpy as np
from utils import load_image_data, create_model, load_latest_model_parameter, save_model_parameter


def test_model(model, x, y, dump_detail):
    for i in range(len(x)):
        x0 = x[i]
        y0 = y[i]
        y1 = model.predict([x0])[0]
        if dump_detail:
            print(i, np.argmax(y0), np.argmax(y1), y1[np.argmax(y1)], y1) 
        else:
            sys.stdout.write('.' if np.argmax(y0) == np.argmax(y1) else 'X')
            if (i + 1) % 100 == 0:
                sys.stdout.write('\r\n')
    sys.stdout.write('\r\n')
        

def main():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--input_image_paths', dest='input_image_paths', default='inputs_128_128_1', nargs='+', help='input image paths, separate by space')
    parser.add_argument('--max_input_count', dest='max_input_count', type=int, default=10000, help='max input image count for train')
    parser.add_argument('--model_parameter_path', dest='model_parameter_path', default='model_parameter')
    parser.add_argument('--dump_detail', dest='dump_detail', default=False, action='store_true')
    
    args = parser.parse_args()
    
    train_x, train_y, test_x, test_y = load_image_data(args.input_image_paths, args.max_input_count) 
    model = create_model(0.0001)
    load_latest_model_parameter(model, args.model_parameter_path)
    test_model(model, train_x, train_y, args.dump_detail)
    test_model(model, test_x, test_y, args.dump_detail)
        
    
if __name__ == '__main__':
    main()