# importing libraries required

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import scipy as sp
import cv2
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.applications import EfficientNetB7
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
import glob
import os 
from keras.layers import *
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop


# reading each GIF and getting frame count statistics

gif_frames_pos = list()
gif_frames_neg = list()

path = '/home/tjbohari/NUIG/Thesis/Datasets/GIF/positive/*.gif'
pos = glob.iglob(path)


path = '/home/tjbohari/NUIG/Thesis/Datasets/GIF/negative/*.gif'
neg = glob.iglob(path)

for i in pos:
    cap = cv2.VideoCapture(i)
    gif_frames_pos.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i in neg:
    cap = cv2.VideoCapture(i)
    gif_frames_neg.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))



print(pd.DataFrame(gif_frames_pos)[0].describe() ,"\n")
print(pd.DataFrame(gif_frames_neg)[0].describe())



seq_len = 300 # number of maximum frames to be considered
frame_count = 16 # numebr of frames to take
img_height, img_width = 112, 112
classes = ["positive", "negative"]
channels = 3

# Generating train and test set from the directories containing GIF files

"""
Below funciton takes list of frame and number of frames needed in each frame list
and pads dummy frames if frame count is less
"""

def frame_padding( frame_list, padding = 40):
    count = 0
    
    dummy_image = np.zeros((padding, img_height, img_width, channels))
    
    frame_count = len(frame_list)

    for _ in range(frame_count):
        dummy_image[count] = frame_list[count]
        count += 1

    return dummy_image

"""
This function returns list of frames from a video path provided as an argument
"""

def frames_extraction(video_path):
    frames_list = []
     
    vidObj = cv2.VideoCapture(video_path)
    # Used as counter variable 
    count = 0
    while count <= seq_len: 
        success, image = vidObj.read() 
        if success:
            if count % 4 == 0:
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (img_height, img_width))
                frames_list.append(image)
            count += 1
        else:
            break
    return np.array(frames_list)

"""
Below function takes a directory where there are subfolders for different categories
and returns frame list as X and category of each GIF for each video
"""
def create_data(input_dir):
    X = []
    Y = []
     
    classes_list = os.listdir(input_dir)
     
    for c in classes_list:
        print(c)
        files_list = os.listdir(os.path.join(input_dir, c))
        for i, f in enumerate(files_list):
            frames = frames_extraction(os.path.join(os.path.join(input_dir, c), f))
            if len(frames) < frame_count:
                frames = frame_padding(frames, 40)
            if len(frames) == frame_count:
                X.append(frames) 

                if c == "negative":
                    Y.append(1)
                else:
                    Y.append(0)
    return X, Y

# creating dataset for GIF models

X_500_3, y_500_3 = create_data("GIF_DataGen/train/")

X = np.array(X_500_3, dtype= "float16")

y = np.array(y_500_3, dtype ="float16")

# splitting data in tarin and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

X_train = X_train / 255
X_test = X_test / 255

# declaring some required shapes

seq_len = 16
img_height, img_width = 112, 112
channels = 3
classes = ["positive", "negative"]

# Conv_LSTM model code


model_convlstm = Sequential()

# adding conv_lstm layers

model_convlstm.add(ConvLSTM2D(filters = 64, kernel_size = (3, 3), return_sequences = True,\
                     activation = "relu",\
                     data_format = "channels_last", input_shape = (seq_len, img_height, img_width, channels)))


model_convlstm.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), return_sequences = False,\
                     activation = "relu"))

model_convlstm.add(Flatten())
model_convlstm.add(Dense(50, activation="relu"))
model_convlstm.add(Dropout(0.5))
model_convlstm.add(Dense(2, activation = "softmax"))


opt = keras.optimizers.Adam()
model_convlstm.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])


model_convlstm.summary()

callb = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy', min_delta=0, patience=3, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)

# training model

history_conv_lstm = model_convlstm.fit(x = X_train, y = y_train, epochs=30, batch_size= 5,\
                    shuffle=True, validation_split = 0.2)

# saving model training history
model_convlstm.save("models/model_conv_lstm_30epochs.h5")

np.save('history/history_conv_lstm.npy',history_conv_lstm.history)

# plotting the Conv_LSTM model architecture 
from keras.utils.vis_utils import plot_model
plot_model(model_convlstm, to_file='model_convlstm.png', show_shapes=True, show_layer_names=True)


history_conv_lstm = np.load('history/history_conv_lstm.npy',allow_pickle='TRUE').item()

# plotting the loss graph of training

plt.plot(history_conv_lstm['loss'])
plt.plot(history_conv_lstm['val_loss'])

plt.title('Loss values for Conv_LSTM Model')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(0, 10)
plt.xticks(np.arange(0, 10+1, 2))
plt.legend(['training', 'validation'], loc='upper right')
plt.show()
plt.savefig("../loss_xception.png")

# plotting the accuracy graph of training

plt.plot(history_conv_lstm['accuracy'])
plt.plot(history_conv_lstm['val_accuracy'])

plt.title('Accuracy values for Conv_LSTM Model')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.xlim(0, 10)
plt.xticks(np.arange(0, 10+1, 2))
plt.legend(['training', 'validation'], loc='best')
plt.show()

# #loading the saved mode
# model_convlstm = keras.models.load_model('models/model_conv_lstm.h5')

# model_convlstm.evaluate(X_train, y_train)

# model_convlstm.evaluate(X_test, y_test)

# code for Conv_3D model trained from scratch

seq_len = 16
img_height, img_width = 112, 112
channels = 3
classes = ["positive", "negative"]
ip = Input(shape=(50,100,100,1))

leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.3)

model_conv3 = Sequential()
model_conv3.add(Conv3D(filters = 128, kernel_size = (3, 3, 3), strides=(1, 1, 1),\
                       activation = leakyrelu,\
                      input_shape = (seq_len, img_height, img_width, channels)))

model_conv3.add(AveragePooling3D(
    pool_size=(1,2, 2), strides=(1,2,2), padding="valid", data_format=None))

model_conv3.add(Conv3D(filters = 128, kernel_size = (3, 3, 3), strides=(1, 1, 1),\
                       activation = leakyrelu))

model_conv3.add(AveragePooling3D(
    pool_size=(1,2, 2), strides=(1,2,2), padding="valid", data_format=None))


model_conv3.add(Conv3D(filters = 128, kernel_size = (3, 3, 3), strides=(1, 1, 1),\
                       activation = leakyrelu))

model_conv3.add(AveragePooling3D(
    pool_size=(1,2, 2), strides=(1,2,2), padding="valid", data_format=None))


model_conv3.add(Conv3D(filters = 128, kernel_size = (3, 3, 3), strides=(1, 1, 1),\
                       activation = leakyrelu))

model_conv3.add(AveragePooling3D(
    pool_size=(1,2, 2), strides=(1,2,2), padding="valid", data_format=None))


model_conv3.add(BatchNormalization())
model_conv3.add(Flatten())

model_conv3.add(Dense(100, activation=leakyrelu))
model_conv3.add(Dropout(0.5))
model_conv3.add(Dense(100, activation=leakyrelu))

model_conv3.add(Dense(2, activation = "softmax"))

# compiling the model 
opt = keras.optimizers.Adam()
model_conv3.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

model_conv3.build(input_shape = (400, seq_len, img_height, img_width, channels))
model_conv3.summary()

# training the model

history_con3d = model_conv3.fit(x = X_train, y = y_train, epochs=30, batch_size= 10,\
                    shuffle=True, validation_split = 0.2)

# saving the model and history of training

model_conv3.save("model/model_conv3d_30epochs.h5")

np.save('history/history_conv3d.npy',history_con3d.history)

# loading the training history for plotting graphs

history_conv3d = np.load('history/history_conv3d.npy',allow_pickle='TRUE').item()

# plotting accuracy graph 

plt.plot(history_con3d.history['accuracy'])
plt.plot(history_con3d.history['val_accuracy'])

plt.title('Accuracy values for Conv_3D Model')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.xlim(0, 30)
plt.xticks(np.arange(0, 30+1, 2))
plt.legend(['training', 'validation'], loc='lower right')
plt.show()

# plotting loss graph

plt.plot(history_con3d.history['loss'])
plt.plot(history_con3d.history['val_loss'])

plt.title('Loss values for Conv_3D Model')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(0, 30)
plt.xticks(np.arange(0, 30+1, 2))
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

#plotting the architecture of model

from keras.utils.vis_utils import plot_model
plot_model(model_conv3, to_file='/content/drive/MyDrive/NUIG/Thesis/model_conv3.png', show_shapes=True, show_layer_names=True)


# importing required libraries

from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers import InputLayer


"""
Below function returns model with architecuture used in sports1m
"""

def get_model(summary=False):
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st layer group
    model.add(InputLayer(input_shape=( 16, 112, 112, 3)))
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', 
                            padding='same', name='conv1',
                            strides=(1, 1, 1), 
                            ))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, (3, 3, 3), activation='relu', 
                            padding='same', name='conv2',
                            strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, (3, 3, 3), activation='relu', 
                            padding='same', name='conv3a',
                            strides=(1, 1, 1)))
    model.add(Convolution3D(256, (3, 3, 3), activation='relu', 
                            padding='same', name='conv3b',
                            strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, (3, 3, 3), activation='relu', 
                            padding='same', name='conv4a',
                            strides=(1, 1, 1)))
    model.add(Convolution3D(512, (3, 3, 3), activation='relu', 
                            padding='same', name='conv4b',
                            strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, (3, 3, 3), activation='relu', 
                            padding='same', name='conv5a',
                            strides=(1, 1, 1)))
    model.add(Convolution3D(512, (3, 3, 3), activation='relu', 
                            padding='same', name='conv5b',
                            strides=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))
    if summary:
        print(model.summary())
    return model

model_conv3d_sports1m = get_model(summary=True)

model_conv3d_sports1m.load_weights('weights_C3D_sports1M_tf.h5')

# setting the training of layers as false

for layer in model_conv3d_sports1m.layers:
    layer.trainable = False

# deleting layers not required

for i in range(8):
  model_conv3d_sports1m.pop()

# adding new layers for training on dataset

leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.3)

model_conv3d_sports1m.add(Convolution3D(512, (3, 3, 3), activation=leakyrelu, 
                            padding='same', name='conv6',
                            strides=(1, 1, 1)))

model_conv3d_sports1m.add(ZeroPadding3D(padding=(0, 1, 1)))
model_conv3d_sports1m.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           padding='valid', name='pool6'))
model_conv3d_sports1m.add(Flatten())

# adding fully connected layers

LeakyReLU = tf.keras.layers.LeakyReLU()
model_conv3d_sports1m.add(Dense(50, activation="relu", name='fc7'))
model_conv3d_sports1m.add(Dropout(.5))
model_conv3d_sports1m.add(Dense(2, activation="relu", name='fc8'))

model.summary()

print(y_train.shape)
print(X_train.shape)

# compiling the model

loss = tf.keras.losses.BinaryCrossentropy()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', mode='min', factor=0.5, patience=3, min_lr=0.000001, verbose=1, cooldown=0)
opt = tf.keras.optimizers.Adam(learning_rate= 0.1)
model_conv3d_sports1m.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

# fitting the model

history_conv3d_sports1m = model_conv3d_sports1m.fit(X_train, y_train, epochs = 30, batch_size= 20, validation_split = 0.2)

# saving model and history of training

model_conv3d_sports1m.save("models/model_conv3d_sports1m_30epochs.h5")

np.save("history/history_conv3d_sports1m.npy", history_conv3d_sports1m.history)

# plotting accuracy graph

plt.plot(history_conv3d_sports1m['accuracy'])
plt.plot(history_conv3d_sports1m['val_accuracy'])

plt.title('Accuracy values for Conv_3D Model with Transfer Learning')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.xlim(0, 30)
plt.xticks(np.arange(0, 30+1, 4))
plt.legend(['training', 'validation'], loc='best')
plt.show()


# plotting loss graph

plt.plot(history_conv3d_sports1m['loss'])
plt.plot(history_conv3d_sports1m['val_loss'])

plt.title('Loss values for Conv_3D Model with Transfer Learning')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(0, 30)
plt.xticks(np.arange(0, 30+1, 4))
plt.legend(['training', 'validation'], loc='center left')
plt.show()

# Evaluating all models with training and test set

model_conv3d_sports1m = keras.models.load_model('models/model_conv3d_sports1m.h5')
model_conv3 = keras.models.load_model("models/model_conv3d.h5")
model_convlstm = keras.models.load_model("models/model_conv_lstm.h5")

print("Training accuracy for Conv3D model", model_conv3.evaluate(X_train, y_train))
print("Test accuracy for Conv3D model", model_conv3.evaluate(X_test, y_test))

print("Training accuracy for ConvLSTM model", model_convlstm.evaluate(X_train, y_train))
print("Test accuracy for ConvLSTM model", model_convlstm.evaluate(X_test, y_test))

print("Training accuracy for Conv3D sports1m model", model_conv3d_sports1m.evaluate(X_train, y_train))
print("Test accuracy for Conv3D sports1m model", model_conv3d_sports1m.evaluate(X_test, y_test))




# ## Image Modal

# importing models from keras applications

from tensorflow.keras.applications import MobileNetV2


## preprocessing of the image and data augmentation using different techniques

## Augmentation applied to the training images for mobilenet model

train_datagen_mob = ImageDataGenerator(
        rescale=1./255, #scaled to range 0 and 1
        shear_range=0.2, # shear transformation
        zoom_range=0.2, #zoom
        horizontal_flip=True) # horizontal flip

## Augment applied to the training images 
test_datagen_mob = ImageDataGenerator(rescale=1./255) ## scaled to range 0 and 1

## reading the images directly from the training directory and augmenting them while the model is training
training_set_mob = train_datagen.flow_from_directory(
        'Img_DataGen/train/',
        target_size=(224, 224),
        batch_size=5,
        class_mode='binary'
        )

## Reading the test images from the directory and scaling is performed on all the images
test_set_mob = test_datagen.flow_from_directory(
        'Img_DataGen/test/',
        target_size=(224, 224),
        batch_size=5,
        class_mode='binary',shuffle=True)


## preprocessing of the image and data augmentation using different techniques

## Augmentation applied to the training images for Xception model
train_datagen = ImageDataGenerator(
        rescale=1./255, #scaled to range 0 and 1
        shear_range=0.2, # shear transformation
        zoom_range=0.2, #zoom
        horizontal_flip=True) # horizontal flip

## Augment applied to the training images 
test_datagen = ImageDataGenerator(rescale=1./255) ## scaled to range 0 and 1

## reading the images directly from the training directory and augmenting them while the model is training
training_set = train_datagen.flow_from_directory(
        'Img_DataGen/train/',
        target_size=(299, 299),
        batch_size=5,
        class_mode='binary'
        )

## Reading the test images from the directory and scaling is performed on all the images
test_set = test_datagen.flow_from_directory(
        'Img_DataGen/test/',
        target_size=(299, 299),
        batch_size=5,
        class_mode='binary',shuffle=True)

# Initiating the MobileNetV2 model 

mobile_base_model = MobileNetV2(input_shape=(224,224,3), ## shape of the image 
                               include_top = False, # last layer has been removed
                               weights = "imagenet")

## the layer in the base model are non-trainable and layer added after this will be trained

for layer in mobile_base_model.layers:
    layer.trainable = False

# adding additional layers to the model

x = mobile_base_model.output 

x = keras.layers.GlobalAveragePooling2D()(x)

dense = keras.layers.Dense(2, activation='softmax')(x) 

mobilenet_model = keras.Model(inputs=mobile_base_model.input, outputs=dense)

mobilenet_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.01),\
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

# fitting the model with training data

mobilenet_history = mobilenet_model.fit(training_set,               
                    steps_per_epoch=80,
                    epochs=20, shuffle = True
                   )

# Evaluating the model performance

print("Training Accuracy: ", mobilenet_model.evaluate(training_set_mob))


print("Testing Accuracy: ", mobilenet_model.evaluate(test_set_mob))


# Plotting the loss and accuracy graphs

plt.plot(mobilenet_history.history['acc'])
plt.title('Model Accuracy for Mobilenet V2')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.xlim(0, 20)
plt.xticks(np.arange(0, 20+1, 4))
plt.show()

plt.savefig("../accuracy_mobilenet.png")


# loss graph

plt.plot(mobilenet_history.history['loss'])
plt.title('Loss values for Mobilenet V2')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(0, 20)
plt.xticks(np.arange(0, 20+1, 4))
plt.show()

plt.savefig("../loss_mobilenet.png")



## Xception model 


xception_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(229, 229, 3),
)

for layer in xception_model.layers:
    layer.trainable = False


xception_x = xception_model.output

# adding additional layers for training

xception_x = keras.layers.GlobalAveragePooling2D()(xception_x)

xception_dense = keras.layers.Dense(2, activation='softmax')(xception_x)

x_ception_model = keras.Model(inputs=xception_model.input, outputs=xception_dense)

x_ception_model.compile(optimizer=RMSprop(lr=0.01), ## learning rate = 0.0001
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

callb = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0, patience=0, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)


x_ception_history =  x_ception_model.fit(training_set, steps_per_epoch=100, epochs= 20)


# plotting accuracy and loss graphs for the model

plt.plot(x_ception_history.history['acc'])
plt.title('Model Accuracy for Xception Model')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.xlim(0, 20)
plt.xticks(np.arange(0, 20+1, 4))
plt.show()

plt.savefig("../accuracy_x_ception.png")


# plotting loss graph

plt.plot(x_ception_history.history['loss'])
plt.title('Loss values for Xception Model')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(0, 20)
plt.xticks(np.arange(0, 20+1, 2))
plt.show()

plt.savefig("../loss_xception.png")

# Evaluating model on training and test set

print("Training Accuracy for Xception model:", x_ception_model.evaluate(training_set))


print("Test Accuracy for Xception model: ", x_ception_model.evaluate(test_set))

# predicting using the model

predictions = x_ception_model.predict(test_set)


x_ception_model.save("models/image_model_xception.h5")


# graph for comparision of accuracies

plt.plot(mobilenet_history.history['acc'])
plt.plot(x_ception_history.history['acc'])
plt.title('Accuracy Comparison for Image models ')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(0, 20)
plt.xticks(np.arange(0, 20+1, 4))
plt.legend(['Mobilenet_V2', 'Xception'], loc='lower right')
plt.show()

plt.savefig("../images/accuracy_comparision_imagemodel.png")





# # Text Modal

#importing the tsv file containing tweet id and sentiment score

""" The file contains tweet ID and score for each sentiment. Max of score is
taken to find the sentiment of the tweet.
"""

text_df = pd.read_csv("text/t4sa_text_sentiment.tsv", sep="\t")

#taking first 5000 records from the file for faster processing
text_df_500 = text_df[0:4999]

# finding the sentiment of tweet by taking the max score from the three categories

category_list = list()
for i in range(4999):
    max_value = max(text_df_500['NEG'][i], text_df_500['NEU'][i], text_df_500['POS'][i])
    if max_value == text_df_500['NEG'][i]:
        category_list.append('NEG')
    elif max_value == text_df_500['NEU'][i]:
        category_list.append('NEU')
    else:
        category_list.append('POS')

# creating a new column of category 

text_df_500 = text_df_500.assign(category = category_list)

# checking the value counts for each category

text_df_500.category.value_counts()

# taking 200 tweeks for each category

text_df_200 = text_df_500.groupby('category').head(200)

# checking count for each category

text_df_200.category.value_counts()

# reading the tweet files

"""
This files contains the tweet id and corresponding tweet
"""

raw_tweets = pd.read_csv('text/raw_tweets_text.csv')


raw_tweets.head(10)


final_text_df = text_df_200.merge(raw_tweets, left_on="TWID", right_on="id")

# cleaning the tweets and spliting them on : and stripping white spaces

final_text_df['text_clean'] = final_text_df['text'].str.split(":", expand=True)[1].str.strip()

final_text_df.head(10)


final_t_df = final_text_df[['text_clean', 'category']]


final_t_df.shape

# importing the VADER SENTIMENT LIBRARY

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# running sentiment analyzer on clean text first

sent_predicted_text = list()
for i in final_t_df['text_clean']:
    res = analyzer.polarity_scores(i)
    sent_predicted_text.append(res['compound'])

# calculating the accuracy

sent_text = list()
for i in sent_predicted_text:
    if i > 0:
        sent_text.append('POS')
    elif i<0:
        sent_text.append('NEG')
    else:
        sent_text.append('NEU')

final_t_df = final_t_df.assign(predicted = sent_text)

counter = 0
for i in range(len(final_t_df)):
    if final_t_df['category'][i] == final_t_df['predicted'][i]:
        counter+= 1

print("Accuracy for clean text is: ", counter /600)

# predicting sentiment on uncleaned text

sent_predicted_text = list()
for i in final_text_df['text']:
    res = analyzer.polarity_scores(i)
    sent_predicted_text.append(res['compound'])


sent_text = list()
for i in sent_predicted_text:
    if i >= 0.05:
        sent_text.append('POS')
    elif i < -0.05:
        sent_text.append('NEG')
    else:
        sent_text.append('NEU')


final_t_df = final_t_df.assign(predicted_notclean = sent_text)

# calculating accuracy

counter = 0
for i in range(len(final_t_df)):
    if final_t_df['category'][i] == final_t_df['predicted_notclean'][i]:
        counter+= 1
print("Accuracy is ", counter / 600)