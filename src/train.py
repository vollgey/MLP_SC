import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

EPOCHS = 10
RADIUS = 0.05

INPUT_PCD_DIR = '/home/soma/development/slope_classification5/dataset/train_' + str(RADIUS) + '/'
OUTPUT_WEIGHT_DIR = '/home/soma/development/slope_classification5/weight/'

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

if __name__ == "__main__":
    ##################
    ## Load file
    ##################
    now = datetime.datetime.now()
    print(str(now))

    _filenames = os.listdir(INPUT_PCD_DIR)
    filenames = [INPUT_PCD_DIR + fname for fname in _filenames]
    filenames = sorted(filenames)

    ##################
    ## Load data
    ##################
    train_clouds = np.empty((0, 7))
    for fname in filenames:
        cloud = np.loadtxt(fname)
        train_clouds = np.vstack((train_clouds, cloud))
    print('total train shape:', train_clouds.shape)

    ##################
    ## Make model
    ##################
    metrics = METRICS

    model = models.Sequential()
    model.add(Dense(2, input_shape=(1, 4), activation='relu'))
    # model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    
    print(model.summary())

    ##################
    ## Train model
    ##################
    x_train = train_clouds[:, 2:6]
    y_train = train_clouds[:, 6]
    x_train = x_train.reshape(-1, 1, 4)
    y_train = y_train.reshape(-1, 1, 1)
    print('x_train.shape:', x_train.shape, 'y_train.shape', y_train.shape)

    # history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=100)
    history = model.fit(x_train, y_train, nb_epoch=EPOCHS, batch_size=100)
    print('Finish training')

    score = model.evaluate(x_train, y_train)

    ##################
    ## Plot fitting
    ##################
    acc = history.history['accuracy']
    # val_acc = history.history['val_acc']
    loss = history.history['loss']
    # val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training loss')
    plt.legend()

    plt.show()

    ##################
    ## Save data
    ##################
    # print('Start save model')
    # out_dir = OUTPUT_WEIGHT_DIR+'/'+str(EPOCHS)+'/'+str(RADIUS)+'/'+'{0:%Y%m%d_%H%M%S}'.format(now)+'/'
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    # model.save(out_dir+'{0:%Y%m%d_%H%M%S}.h5'.format(now))

    # slope_iou = score[1] / (score[1] + score[2] + score[4])
    # ground_iou = score[3] / (score[3] + score[2] + score[4])
    # mean_iou = (slope_iou + ground_iou) / 2
    # score = np.append(score, [slope_iou, ground_iou, mean_iou])
    # np.set_printoptions(suppress=True)
    # print(score)
    
    # csv_file = open(out_dir+'{0:%Y%m%d_%H%M%S}.csv'.format(now), 'w')
    # writer = csv.writer(csv_file)
    # writer.writerow(score)
