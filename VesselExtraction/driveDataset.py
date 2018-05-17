from PIL import Image
import numpy as np
import cv2

def loadImages(mode):
    xdir = 'DRIVE/' + mode + '/images/'
    xext = '_' + mode + '.tif'
    ydir = 'DRIVE/' + mode + '/1st_manual/'
    yext = '_manual1.gif'

    x = np.empty((0, 584, 565, 3))
    y = np.empty((0, 584, 565))
    for i in range(0, 20):
        if mode == 'training':
            idx = i + 21
        else:
            idx = i + 1
        imx = cv2.imread(xdir + '{:02d}'.format(idx) + xext)
        imx = cv2.normalize(imx.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        arrx = np.asarray(imx)# [..., ::-1] #BGR to RGB if required

        imy = Image.open(ydir + '{:02d}'.format(idx) + yext)
        arry = np.asarray(imy)
        arry = cv2.normalize(arry.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        arrx = arrx.reshape((1, 584, 565, 3))
        arry = arry.reshape((1, 584, 565))
        x = np.append(x, arrx, axis=0)
        y = np.append(y, arry, axis=0)

    return x, y


def saveImages(y_test, y_pred):
    for i in range(0, 20):
        imtest = cv2.normalize(y_test[i, :, :], None, 0, 255, cv2.NORM_MINMAX)
        impred = cv2.normalize(y_pred[i, :, :], None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite('DRIVE/out/pred/' + '{:02d}'.format(i+1) + '_pred.jpg', impred)
        diff = np.zeros((584, 565, 3))
        ret, tpred = cv2.threshold(impred, 127, 255, cv2.THRESH_BINARY)
        diff[:, :, 0] = tpred
        diff[:, :, 1] = imtest
        cv2.imwrite('DRIVE/out/diff/' + '{:02d}'.format(i + 1) + '_diff.jpg', diff)

