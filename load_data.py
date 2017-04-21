import numpy as np
import cv2
import os.path as path
from utils import imagenet_mean_bgr


def convert_from_relative_to_absolute(h, w, x_min, y_min, x_max, y_max):
    """
    Convert from relative coordinates (range 0, 1) to absolute coordinates given a frame (range h, w)

    Parameters
    ----------
    h : int
        Image height
    w : int
        Image width
    x_min : float
        X coordinate of top-left corner (in range 0, 1)
    y_min : float
        Y coordinate of top-left corner (in range 0, 1)
    x_max : float
        X coordinate of bottom-right corner (in range 0, 1)
    y_max : float
        Y coordinate of bottom-right corner (in range 0, 1)

    Returns
    -------
    coords : list
        Input coordinates casted to image size -> range (0, h) and (0, w)
    """
    x_min = x_min * w
    y_min = y_min * h
    x_max = x_max * w
    y_max = y_max * h
    return map(np.int32, [x_min, y_min, x_max, y_max])


def extract_crop(frame, x_min, y_min, x_max, y_max):
    """
    Extract vehicle crop from the image.
    Crop is resized to 224x224 which is ResNet input size.

    Parameters
    ----------
    frame : ndarray
        Image to process
    x_min : float
        X coordinate of top-left corner (in range 0, 1)
    y_min : float
        Y coordinate of top-left corner (in range 0, 1)
    x_max : float
        X coordinate of bottom-right corner (in range 0, 1)
    y_max : float
        Y coordinate of bottom-right corner (in range 0, 1)

    Returns
    -------
    crop : ndarray
        Crop containing vehicle, resized to 224x224 pixel
    """
    h, w = frame.shape[:2]

    x_min, y_min, x_max, y_max = convert_from_relative_to_absolute(h, w, x_min, y_min, x_max, y_max)

    # extract crop from frame
    crop = frame[y_min:y_max, x_min:x_max, :].copy()

    crop = cv2.resize(crop, (224, 224))

    return crop


def get_sample_batch(data_dir):
    """
    Load sample data useful for model "hello world".
    """
    X_coords, X_images, X_crops, X_images_original = [], [], [], []
    Y_coords, Y_images, Y_crops, Y_dist, Y_yaw = [], [], [], [], []

    with open(path.join(data_dir,'sample_data.txt'), 'rb') as f:
        logs = f.readlines()

        for log in logs:

            # retrieve line values
            log = log.strip().split(',')

            # parse a log line
            frame_f, frame_b = log[:2]
            bbox_id, bbox_model = log[2:4]
            bbox_dist, bbox_yaw = map(np.float32, log[4:6])
            coords_frontal = map(np.float32, log[6:10])
            coords_birdeye = map(np.float32, log[10:])

            # load images
            frame_frontal_path = path.join(data_dir, frame_f.strip())
            frame_birdeye_path = path.join(data_dir, frame_b.strip())
            if not path.exists(frame_frontal_path) or not path.exists(frame_birdeye_path): continue
            frame_frontal = cv2.imread(frame_frontal_path, cv2.IMREAD_COLOR)
            frame_birdeye = cv2.imread(frame_birdeye_path, cv2.IMREAD_COLOR)

            # extract crops from whole frames
            crop_frontal = extract_crop(frame_frontal, *coords_frontal)
            crop_birdeye = extract_crop(frame_birdeye, *coords_birdeye)

            if crop_frontal is not None and crop_birdeye is not None:

                # convert from (0, 1) to tanh range (-1, 1)
                coords_birdeye = [2 * (c - 0.5) for c in coords_birdeye]

                # append all needed stuff to output structures
                X_coords.append(coords_frontal)  # append frontal coords
                X_crops.append(crop_frontal)  # append frontal crops
                X_images.append(frame_frontal)  # append frontal frames
                X_images_original.append(frame_frontal)  # append frontal frames
                Y_coords.append(coords_birdeye)  # append birdeye coords
                Y_crops.append(crop_birdeye)  # append birdeye crops
                Y_images.append(frame_birdeye)  # append birdeye frames
                Y_dist.append(bbox_dist)  # append bbox distance
                Y_yaw.append(bbox_yaw)  # append bbox yaw

    # preprocess X crops by subtracting mean and put channels first
    for b in range(0, len(X_coords)):
        X_crops[b] = imagenet_mean_bgr(frame_bgr=X_crops[b], op='subtract').transpose(2, 0, 1)

    # convert all stuff to ndarray
    X_coords, Y_coords = np.array(X_coords), np.array(Y_coords)
    X_crops, Y_crops = np.array(X_crops), np.array(Y_crops)
    X_images, Y_images = np.array(X_images), np.array(Y_images)
    Y_dist, Y_yaw = np.array(Y_dist), np.array(Y_yaw)
    X_images_original = np.array(X_images_original)

    return X_coords, X_crops, X_images, X_images_original, Y_coords, Y_crops, Y_images, Y_dist, Y_yaw

