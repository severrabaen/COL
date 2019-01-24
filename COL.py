#!pip install icrawler
#↑ install first!


#import
import tensorflow as tf
from icrawler.builtin import GoogleImageCrawler
from google.colab import files
import numpy as np
from PIL import Image
import sys, os
import os, glob
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import scipy
import scipy.misc

#declaration
classes = ["cabeges","lettuces"]
num_classes = len(classes)
image_size = 50
num_testdata = 25
imsize=(50,50)
X_train = []
X_test  = []
Y_train = []
Y_test  = []


#get images from Google search
crawler = GoogleImageCrawler(storage={"root_dir": "cabeges"})
crawler.crawl(keyword="キャベツ", max_num=100)
crawler = GoogleImageCrawler(storage={"root_dir": "lettuces"})
crawler.crawl(keyword="レタス", max_num=100)

#classify
for index, classlabel in enumerate(classes):
  photos_dir="./"+classlabel
  files=glob.glob(photos_dir+"/*.jpg")
  for i, file in enumerate(files):
    image=Image.open(file)
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    if i < num_testdata:
        X_test.append(data)
        Y_test.append(index)
    else:
        for angle in range(-20, 20, 5):
            img_r = image.rotate(angle)
            data = np.asarray(img_r)
            X_train.append(data)
            Y_train.append(index)
            img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
            data = np.asarray(img_trans)
            X_train.append(data)
            Y_train.append(index)
    
X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(Y_train)
y_test  = np.array(Y_test)
mon=(X_train, X_test, y_train, y_test)
#save npy file
np.save("./classified.npy",mon)

#import data
def load_data():
    X_train, X_test, y_train, y_test = np.load("./classified.npy")
    X_train = X_train.astype("float") / 256
    X_test  = X_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test  = np_utils.to_categorical(y_test, num_classes)
    return X_train, y_train, X_test, y_test
  
#train
def train(X, y, X_test, y_test):
    model = Sequential()
    model.add(Conv2D(32,(3,3), padding='same',input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.00005, decay=1e-6)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.fit(X, y, batch_size=28, epochs=40)
    model.save('./cnn.h5')

    return model
  
  
def main():
    #input
    X_train, y_train, X_test, y_test = load_data()
    #learn
    model = train(X_train, y_train, X_test, y_test)
    
    
main()

testpic     = "./exc.png"
keras_param = "./cnn.h5"

def load_image(path):
    img = scipy.misc.imread(path, mode="RGB")
    img = scipy.misc.imresize(img, imsize)
    img = img / 255.0
    return img

def get_file(dir_path):
    filenames = os.listdir(dir_path)
    return filenames

if __name__ == "__main__":
  
    model = load_model('cnn.h5')
    img = load_image(testpic)
    prd = model.predict(np.array([img]))
    print(prd)
    prelabel = np.argmax(prd, axis=1)
    if prelabel == 0:
        print("それはキャベツ")
    elif prelabel == 1:
        print("それはレタス")
