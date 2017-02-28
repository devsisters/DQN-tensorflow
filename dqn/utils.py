import time
import numpy as np
import tensorflow as tf
import sys
if (sys.version_info[0]==2):
  import cPickle
elif (sys.version_info[0]==3):
  import _pickle as cPickle

try:
  from scipy.misc import imresize
except:
  import cv2
  imresize = cv2.resize

def rgb2gray(image):
  return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def timeit(f):
  def timed(*args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()

    print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
    return result
  return timed

def get_time():
  return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

@timeit
def save_pkl(obj, path):
  with open(path, 'w') as f:
    cPickle.dump(obj, f)
    print("  [*] save %s" % path)

@timeit
def load_pkl(path):
  with open(path) as f:
    obj = cPickle.load(f)
    print("  [*] load %s" % path)
    return obj

@timeit
def save_npy(obj, path):
  np.save(path, obj)
  print("  [*] save %s" % path)

@timeit
def load_npy(path):
  obj = np.load(path)
  print("  [*] load %s" % path)
  return obj
