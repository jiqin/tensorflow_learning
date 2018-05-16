import cv2
import argparse
from utils import save_image


def capture_image(image_root, hand_gesture):
    cap = cv2.VideoCapture(0)
    
    image_id = 0
    while (True):
        ret, image = cap.read()
        cv2.imshow("capture", image)
        save_image(image, image_id, image_root, hand_gesture)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        image_id += 1
        print('image_id:', image_id)
    cap.release()
    cv2.destroyAllWindows()
        

def main():
    parser = argparse.ArgumentParser(description='Capture')
    parser.add_argument('--image_root', dest='image_root', default='inputs_128_128', help='saved image root')
    parser.add_argument('--hand_gesture', dest='hand_gesture', default='scissor', help='option: rock, paper, scissor')
    args = parser.parse_args()
    capture_image(args.image_root, args.hand_gesture)

    
if __name__ == '__main__':
    main()
