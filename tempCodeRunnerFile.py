import os
import shutil
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from functions_training.check_files_is_good import checkFileIsGood
from functions_training.make_dir_train_and_validation import makeDirTrainAndValidation
from sklearn.utils import class_weight