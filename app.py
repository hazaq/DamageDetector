import os
import sys
from flask import Flask, request, render_template, url_for, redirect, send_from_directory

import tensorflow as tf 
from os import listdir
from os.path import isfile, join

import skimage
import skimage.io 
import skimage.transform
import numpy as np
import cv2

ROOT_DIR = os.getcwd()
MRCNN = os.path.join(ROOT_DIR, 'mrcnn/')
sys.path.append(MRCNN)
from config import Config
import utils
import model as modellib
from mrcnn import visualize

#model_side = tf.keras.models.load_model('model/car-sides.h5')

class Damageconfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Damage"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + baloon
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8

config = Damageconfig()
model_damage = modellib.MaskRCNN(mode="inference", config=config, model_dir="./logs")
model_damage.load_weights("./model/mask_rcnn_damage_0010.h5", by_name=True)

model_damage.keras_model._make_predict_function()

app = Flask(__name__, static_url_path='')

@app.route("/", methods=['GET','POST'])
def CarSider():
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '':            
            photo.save(os.path.join('./images/', photo.filename))

        img = skimage.io.imread(os.path.join('./images/', photo.filename))
#        img = skimage.transform.resize(img, (150,150))
        r = model_damage.detect([img], verbose=1)[0]
        mask = r['masks']
        mask_sum = 0
        damage_area = img * 0
        is_damage = 0
        for i in range(mask.shape[2]):
            mask_sum = (np.sum(mask, -1, keepdims=True) >= 1 )
            is_damage = mask_sum[:,:,0].sum()

        result=''
        if is_damage > 1: 
            result='There is damage'
            damage_area[:,:,0] = mask_sum[:,:,0] * 255
            damage_area[:,:,1] = 0
            damage_area[:,:,2] = 0
            damage_car = cv2.add(img, damage_area)
            axis = r['rois']
            for i in range(axis.shape[0]):
                damage_car = cv2.rectangle(damage_car, (axis[i][1], axis[i][0]), (axis[i][3], axis[i][2]), (0,255,0), 2)

            skimage.io.imsave('./images/damage' + photo.filename, damage_car)
            photo.filename = 'damage' + photo.filename
            confidence = r['scores']
            
        else:
            result='No damage found'
            confidence = 1

        render_data={'file':photo.filename, 'result':result, 'confidence':confidence}
        return render_template('index.html', data=render_data)
    else:
        return render_template('index.html')


@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory('images', path)


if __name__ == '__main__':
    app.run()     
