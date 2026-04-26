import cv2
import numpy as np
import json
import torch

W, H = 640, 480
F_X, F_Y = 686, 686
C_X, C_Y = W // 2, H // 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def project_2d(x, y, z):
    u = (F_X * (x / z) + C_X) / W
    v = (F_Y * (y / z) + C_Y) / H
    return (u, v)

def convert(obj):
    gt = {}
    cam = obj['cameras']['cam0']

    # gaze vector [3]
    gx, gy, gz, _ = obj['eye_details']['look_vec']
    gt['gaze_vector'] = torch.tensor([gx, gy, gz], dtype=torch.float32)

    # pupil 2d position + diameter [3], matching pupil head out_dim=3
    ix, iy, iz = cam['ground_truth']['iris_center']
    pu, pv = project_2d(ix, iy, iz)
    pupil_size = obj['eye_details']['pupil_size']
    iris = cam['iris_2d']
    iris_pts = np.array([(p[0], p[1]) for p in iris], dtype=np.float32)
    (cx, cy), (axis_w, axis_h), angle = cv2.fitEllipse(iris_pts)
    iris_diameter = float(np.sqrt(axis_w * axis_h))
    gt['pupil'] = torch.tensor([pu, pv, iris_diameter * pupil_size], dtype=torch.float32)

    # eyelid shape: 4 upper + 4 lower 2d points flattened to [16]
    upper = [(pt['pos'][0], pt['pos'][1]) for pt in cam['upper_interior_margin_2d']][::2]
    lower = [(pt['pos'][0], pt['pos'][1]) for pt in cam['lower_interior_margin_2d']][::2]
    gt['eyelid_shape'] = torch.tensor(upper + lower, dtype=torch.float32).flatten()

    gt['openness'] = torch.tensor(cam['openness'], dtype=torch.float32)

    return gt