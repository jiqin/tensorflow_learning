import cv2
import time
import argparse
import os
import numpy as np

IMAGE_ROOT = 'images'

def save_image(gray_image, image_id, folder):
        cv2.imwrite(os.path.join(folder, 'image_{0}.jpg'.format(image_id)), gray_image)
        #np.save(os.path.join(folder, 'image_{0}'.format(image_id)), gray_image)

                
def capture_image(hand_gesture):
        root_path = os.path.join(IMAGE_ROOT, hand_gesture)
        os.makedirs(root_path, exist_ok=True)
                
        cap = cv2.VideoCapture(0)
        
        image_id = 0
        while (True):
                ret, frame = cap.read()
                cv2.imshow("capture", frame)
                save_image(frame, image_id, root_path)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                image_id += 1
                #time.sleep(1)
        cap.release()
        cv2.destroyAllWindows()
                

def main():
        parser = argparse.ArgumentParser(description='Capture')
        parser.add_argument('--hand_gesture', dest='hand_gesture', default='scissor', help='option: rock, paper, scissor')
        args = parser.parse_args()
        capture_image(args.hand_gesture)

        
if __name__ == '__main__':
        main()
