import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from keras.utils import CustomObjectScope

from keras import backend as K

H = 256
W = 256


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)





class backgroundRemovalModelUNET():

    def __init__(self):
        self.model = None
        self.model_init()

    def model_init(self):
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            self.model = tf.keras.models.load_model("models/unet.h5")

    def pred(self,frame):


        h, w, _ = frame.shape
        ori_frame = frame
        frame = cv2.resize(frame, (W, H))
        frame = np.expand_dims(frame, axis=0)
        frame = frame / 255.0

        mask = self.model.predict(frame)[0]
        mask = cv2.resize(mask, (w, h))
        mask = mask > 0.5
        mask = mask.astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        photo_mask = mask
        background_mask = np.abs(1 - mask)

        masked_frame = ori_frame * photo_mask

        background_mask = np.concatenate([background_mask, background_mask, background_mask], axis=-1)
        background_mask = background_mask * [0, 0, 255]
        final_frame = masked_frame + background_mask
        final_frame = final_frame.astype(np.uint8)

        return  final_frame








