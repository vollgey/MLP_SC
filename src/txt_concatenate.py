import os
import numpy as np

BASE_DIR = '/home/soma/development/slope_classification6/eme/'
TYPE = '0.5'

if __name__ == "__main__":
    ground_dir = os.path.join(BASE_DIR, 'ground/')
    slope_dir = os.path.join(BASE_DIR, 'slope/')

    out_dir = os.path.join(BASE_DIR, 'test_'+TYPE+'/')

    _ground_f = os.listdir(ground_dir)
    ground_f = [ground_dir + fname for fname in _ground_f]
    ground_f = sorted(ground_f)

    _slope_f = os.listdir(slope_dir)
    slope_f = [slope_dir + fname for fname in _slope_f]
    slope_f = sorted(slope_f)

    for i in range(len(ground_f)):
        ground = np.loadtxt(ground_f[i])
        slope = np.loadtxt(slope_f[i])

        merge = np.concatenate([ground, slope])
        print(merge.shape)

        np.savetxt(out_dir+str(i+1)+'.txt', merge)
    