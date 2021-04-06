from PIL import Image
import os, sys
import numpy
import cv2
from skimage.io import imsave


def load_tiff_data(dataFile, dtype='float32'):
    """
    Loads data from a multilayer .tif file.  
    Returns result as a 3d numpy tensor.
    """
    if not os.path.isfile(dataFile):
        raise RuntimeError('could not find "%s"' % dataFile)

    # load the data from multi-layer TIF files
    dataImg = Image.open(dataFile)
    X = []
    for ii in range(sys.maxsize):
        Xi = numpy.array(dataImg, dtype=dtype)
        X.append(Xi)
        try:
            dataImg.seek(dataImg.tell() + 1)
        except EOFError:
            break

    X = numpy.dstack(X).transpose((2, 0, 1))
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
    return X

def loadDataFromDir(path):
    files = os.listdir(path)
    imgs = []
    for filename in files:
        I = Image.open(os.path.join(path,filename))
        imgs.append(numpy.array(I))
    imgs = numpy.dstack(imgs).transpose((2,0,1))
    imgs = imgs.reshape(imgs.shape[0],imgs.shape[1],imgs.shape[2],1)
    return imgs


if __name__ == '__main__':

    data_file = "D:\keras\liuj\mitochondria\\training.tif"
    x_train = load_tiff_data(data_file, dtype=numpy.uint8)
    for i in range(x_train.shape[0]):
        test = x_train[i, :, :, 0]
        imsave('./images/'+str(i+1).zfill(3)+'.png',test)
        # cv2.imwrite('./masks/neurons/'+str(i+1).zfill(2)+'.jpg',test,[cv2.IMWRITE_JPEG_QUALITY,90])
        # img = cv2.imread('./masks/neurons/'+str(i+1).zfill(2)+'.png')
        # Image.fromarray(test).save('./masks/neurons/'+str(i+1).zfill(2)+'.jpg')


