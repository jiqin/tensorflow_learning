# tensorflow_learning

# create train data
python create_gesture_model_input.py --image_root test_images --hand_gesture rock

# train model
python train_gesture_model.py --input_image_paths test_images --epoch_num 5 --learning_rate 0.0001

# test model
python test_gesture_model.py 
