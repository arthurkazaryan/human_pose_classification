import sys
import mediapipe as mp
import numpy as np
from pathlib import Path
from dataset import split_data
from train import TrainNN
from PIL import Image


def predict(image_path, model):

    pose = mp.solutions.pose.Pose()
    img = np.array(Image.open(image_path), dtype='uint8')
    results = pose.process(img)
    x_data = []
    full_height_points = []
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if lm.visibility > 0.5:
                coords = [lm.x, lm.y, lm.z]
            else:
                coords = [0, 0, 0]
            x_data.append(coords)
            if id in range(1, 11):
                full_height_points.append(lm.visibility)
            elif id in range(27, 33):
                full_height_points.append(lm.visibility)
    pose.reset()
    if not x_data:
        return 'Prediction failed.'

    prediction = ''
    prediction += f'{split_data[np.argmax(model.predict(np.expand_dims(np.array(x_data), 0)))]} view.'
    if np.mean(full_height_points) > 0.9:
        prediction += ' full height.'
    return prediction


model = TrainNN.make_model()
model.load_weights(Path.cwd().joinpath('weights', 'weights.h5'))
predict_answer = predict(sys.argv[1], model)
print(predict_answer)
