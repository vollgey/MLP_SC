import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

EPOCHS = 10
RADIUS = 0.05
MODEL_NAME = '20210123_151740'

INPUT_PCD_DIR = '/home/soma/development/slope_classification6/dataset/test_'+str(RADIUS)+'/'
OUTPUT_PCD_DIR = '/home/soma/development/slope_classification6/output/'
INPUT_WEIGHT_DIR = '/home/soma/development/slope_classification5/weight/'+str(EPOCHS)+'/'+str(RADIUS)+'/'+MODEL_NAME+'/'+MODEL_NAME+'.h5'



if __name__ == "__main__":
    ####################
    ## Load file, model
    ####################
    _filenames = os.listdir(INPUT_PCD_DIR)
    filenames = [INPUT_PCD_DIR + fname for fname in _filenames]

    predicted_ground = np.empty((0, 6))
    predicted_slope = np.empty((0, 6))

    ##################
    ## Load data
    ##################
    test_clouds = np.empty((0, 7))
    point_count = [0]
    for fname in filenames:
        cloud = np.loadtxt(fname)
        point_count = np.append(point_count,cloud.shape[0])
        test_clouds = np.vstack((test_clouds, cloud))
    print('total train shape:', test_clouds.shape)
    print(point_count)

    x_test = test_clouds[:, 2:6]
    y_test = test_clouds[:, 6]
    x_test = x_test.reshape(-1, 1, 4)
    y_test = y_test.reshape(-1, 1, 1)
    print('x_train.shape:', x_test.shape, 'y_train.shape', y_test.shape)


    model = models.load_model(INPUT_WEIGHT_DIR)
    print(model.summary())
    ##################
    ## Predict
    ##################
    raw_predictions = model.predict(x_test)
    print('raw_predictions', raw_predictions.shape)
    predictions = np.round(raw_predictions)
    print('predictions', predictions.shape)

    for j in range(test_clouds.shape[0]):
        if predictions[j,0] == 0:
            predicted_ground = np.vstack(
                (predicted_ground, test_clouds[j, :6]))
        else:
            predicted_slope = np.vstack(
                (predicted_slope, test_clouds[j, :6]))


    print('ground', predicted_ground.shape, 'slope', predicted_slope.shape)

    score = model.evaluate(x_test, y_test)
    print(score)

    ##################
    ## Calc IOU
    ##################
    slope_iou = score[1] / (score[1] + score[2] + score[4])
    ground_iou = score[3] / (score[3] + score[2] + score[4])
    mean_iou = (slope_iou + ground_iou) / 2
    score = np.append(score, [slope_iou, ground_iou, mean_iou])
    np.set_printoptions(suppress=True)
    print(score)

    ##################
    ## Export csv, pcd
    ##################
    # print('Start save model')
    # out_dir = OUTPUT_PCD_DIR+'/'+str(EPOCHS)+'/'+str(RADIUS)+'/'+MODEL_NAME+'/'
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    
    # csv_file = open(out_dir + 'valid_'+MODEL_NAME+'.csv', 'w')
    # writer = csv.writer(csv_file)
    # writer.writerow(score)

    # output_file_name = str(EPOCHS)+'_'+str(RADIUS)+'_'+MODEL_NAME
    # np.savetxt(out_dir + output_file_name + '_ground.txt', predicted_ground)
    # np.savetxt(out_dir + output_file_name + '_slope.txt', predicted_slope)
    # print('successed export')
 