import argparse
import numpy as np
import cv2
from utils import normalize_images, create_model, load_latest_model_parameter, GESTURE_TYPE


def test_model(model):
    cap = cv2.VideoCapture(0)
    while (True):
        ret, image = cap.read()
        cv2.imshow("capture", image)
        ret = model.predict(normalize_images([image]))[0]
        ret_index = np.argmax(ret)
        print(ret_index, GESTURE_TYPE[ret_index], ret[ret_index], ret)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--model_parameter_path', dest='model_parameter_path', default='model_parameter')
      
    args = parser.parse_args()
    
    model = create_model(0.0001)
    load_latest_model_parameter(model, args.model_parameter_path)
    test_model(model)
        
    
if __name__ == '__main__':
    main()