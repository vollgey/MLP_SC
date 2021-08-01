import os
import csv
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

EPOCHS = 10
RADIUS = 0.5

BASE_DIR = '/home/soma/development/slope_classification6/'

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.MeanIoU(name='mean_iou', num_classes=2),
]


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)*2-1
    return result

if __name__ == "__main__":
    train_dir = os.path.join(BASE_DIR, 'dataset/train_'+str(RADIUS)+'/')
    # train_dir = os.path.join(BASE_DIR, 'dataset/check/')
    test_dir = os.path.join(BASE_DIR, 'dataset/test_'+str(RADIUS)+'/')
    weight_dir = os.path.join(BASE_DIR, 'weight/')
    output_dir = os.path.join(BASE_DIR, 'output/')
    now = datetime.datetime.now()
    print(str(now))

    output_dir = output_dir+'/'+str(EPOCHS)+'/'+str(RADIUS)+'/'+'{0:%Y%m%d_%H%M%S}.h5'.format(now)+'/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


###  TRAIN  ###
    ##################
    ## Load file
    ##################
    _filenames = os.listdir(train_dir)
    filenames = [train_dir + fname for fname in _filenames]
    filenames = sorted(filenames)
    print(len(filenames))

    ##################
    ## Set train data
    ##################
    train_clouds = np.empty((0, 7))
    for fname in filenames:
        cloud = np.loadtxt(fname)
        train_clouds = np.vstack((train_clouds, cloud))
    print('total train shape:', train_clouds.shape)

    x_train = train_clouds[:, 2:6]
    y_train = train_clouds[:, 6]
    print(x_train)
    #normalize
    x_train[:,0] = min_max(x_train[:,0], axis=0)

    x_train = x_train.reshape(-1, 1, 4)
    y_train = y_train.reshape(-1, 1, 1)
    print('x_train.shape:', x_train.shape, 'y_train.shape', y_train.shape)
    print(x_train)

    ##################
    ## Make model
    ##################
    metrics = METRICS

    model = models.Sequential()
    model.add(Dense(16, input_shape=(1, 4), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    print(model.summary())

    ##################
    ## Train model
    ##################
    history = model.fit(x_train, y_train, nb_epoch=EPOCHS, batch_size=100)
    print('Finish training')
    train_score = model.evaluate(x_train, y_train)

    ##################
    ## Calc IOU
    ##################
    slope_iou = train_score[1] / (train_score[1] + train_score[2] + train_score[4])
    ground_iou = train_score[3] / (train_score[3] + train_score[2] + train_score[4])
    mean_iou = (slope_iou + ground_iou) / 2
    train_score = np.append(train_score, [slope_iou, ground_iou, mean_iou])
    np.set_printoptions(suppress=True)
    print(train_score)

    ##################
    ## Save data
    ##################
    print('Start save model')
    weight_dir = weight_dir+'/'+str(EPOCHS)+'/'+str(RADIUS)+'/'+'{0:%Y%m%d_%H%M%S}'.format(now)+'/'
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    model.save(weight_dir+'{0:%Y%m%d_%H%M%S}.h5'.format(now))
    
    csv_file = open(weight_dir+'{0:%Y%m%d_%H%M%S}.csv'.format(now), 'w')
    writer = csv.writer(csv_file)
    writer.writerow(train_score)

###  TEST  ###
    ####################
    ## Load file
    ####################
    _filenames = os.listdir(test_dir)
    filenames = [test_dir + fname for fname in _filenames]
    filenames = sorted(filenames)

    ##################
    ## Set test data
    ##################
    test_clouds = np.empty((0, 7))
    point_count = [0]
    counter = 0
    for fname in filenames:
        cloud = np.loadtxt(fname)
        counter += cloud.shape[0]
        point_count = np.append(point_count,counter)
        test_clouds = np.vstack((test_clouds, cloud))
    print('total test shape:', test_clouds.shape)
    print('total count shape:', point_count)

    x_test = test_clouds[:, 2:6]
    y_test = test_clouds[:, 6]

    x_test[:,0] = min_max(x_test[:,0], axis=0)

    x_test = x_test.reshape(-1, 1, 4)
    y_test = y_test.reshape(-1, 1, 1)
    print('x_test.shape:', x_test.shape, 'y_test.shape', y_test.shape)

    ##################
    ## Predict
    ##################
    t1 = time.time()
    raw_predictions = model.predict(x_test)
    t2 = time.time()
    pred_time = t2 - t1
    print('raw_predictions', raw_predictions.shape)
    predictions = np.round(raw_predictions)
    print('predictions', predictions.shape)

    print(point_count)
    for i in range(len(filenames)):
        predicted_ground = np.empty((0, 6))
        predicted_slope = np.empty((0, 6))
        _test_clouds = test_clouds[point_count[i]:point_count[i+1],:]
        _predictions = predictions[point_count[i]:point_count[i+1],:]
        print('_test',_test_clouds.shape,'_pred',_predictions.shape)
        for j in range(_test_clouds.shape[0]):
            if _predictions[j,0] == 0:
                predicted_ground = np.vstack(
                    (predicted_ground, _test_clouds[j, :6]))
            else:
                predicted_slope = np.vstack(
                    (predicted_slope, _test_clouds[j, :6]))

        output_file_name = str(EPOCHS)+'_'+str(RADIUS)+'_'+ str(i+1)
        np.savetxt(output_dir + output_file_name + '_ground.txt', predicted_ground)
        np.savetxt(output_dir + output_file_name + '_slope.txt', predicted_slope)

        print('ground', predicted_ground.shape, 'slope', predicted_slope.shape)
    t3 = time.time()
    test_score = model.evaluate(x_test, y_test)
    t4 = time.time()
    eval_time = t4 - t3
    print(test_score)

    ##################
    ## Calc IOU
    ##################
    slope_iou = test_score[1] / (test_score[1] + test_score[2] + test_score[4])
    ground_iou = test_score[3] / (test_score[3] + test_score[2] + test_score[4])
    mean_iou = (slope_iou + ground_iou) / 2
    test_score = np.append(test_score, [slope_iou, ground_iou, mean_iou])
    test_score = np.append(test_score, [pred_time, eval_time])
    np.set_printoptions(suppress=True)
    print(test_score)

    ##################
    ## Export csv, pcd
    ##################
    print('Start save model')
    
    csv_file = open(output_dir + 'valid_'+'{0:%Y%m%d_%H%M%S}.h5'.format(now)+'.csv', 'w')
    writer = csv.writer(csv_file)
    writer.writerow(test_score)
    print('successed export')