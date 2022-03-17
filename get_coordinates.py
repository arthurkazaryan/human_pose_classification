import zipfile
import pandas as pd
import csv
import tqdm
import mediapipe as mp
import numpy as np
from PIL import Image
from pathlib import Path


# Extracting img.zip. Archive should be placed in a root directory.
with zipfile.ZipFile(Path.cwd().joinpath('img.zip'), 'r') as zip_ref:
    zip_ref.extractall(Path.cwd())


# Csv file should be placed in a root directory.
df = pd.read_csv(Path.cwd().joinpath('df.csv'), index_col=0)

dataframe_rows = ['image_path', 'positions', 'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
                  'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear',
                  'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
                  'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
                  'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
                  'right_index', 'left_thumb', 'right_thumb', 'left_hip',
                  'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                  'right_ankle', 'left_heel', 'right_heel', 'left_foot_index',
                  'right_foot_index']


# Creating a csv-file with [X, Y, Z] coordinates of each body part
dataframe_path = Path.cwd().joinpath('dataset', 'dataframe_coords.csv')
with open(dataframe_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(dataframe_rows)

pose = mp.solutions.pose.Pose()
for i in tqdm.tqdm(range(len(df))):
    image_path = df.iloc[i, 0]
    positions = df.iloc[i, 1]
    img = np.array(Image.open(image_path), dtype='uint8')
    results = pose.process(img)
    if results.pose_landmarks:
        dataframe_dict = {x: '' for x in dataframe_rows}
        dataframe_dict['image_path'] = image_path
        dataframe_dict['positions'] = positions
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if lm.visibility > 0.5:
                coords = [lm.x, lm.y, lm.z]
            else:
                coords = [0, 0, 0]
            dataframe_dict[dataframe_rows[id+2]] = coords

        with open(dataframe_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(list(dataframe_dict.values()))
    pose.reset()
