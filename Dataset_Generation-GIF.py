# importing required libraries
import pandas as pd

# printing all the metrics information which contains metric and corresponding mID

metrics = pd.read_json("metrics.json")

print(metrics)


categories = ["angry", "disgust", "fear", "happiness", "sadness", "surprise"]

# reading json files

# negative sentiments
angry = pd.read_json("GIF/angry.json", lines=True)
disgust = pd.read_json("GIF/disgust.json", lines=True)

# positive sentiments
happiness = pd.read_json("GIF/happiness.json", lines=True)
surprise = pd.read_json("GIF/surprise.json", lines=True)


# downloading the files and placing them in respective folders

import requests

for i, _ in enumerate(angry["content_data"]):
    link = angry["content_data"][i]["embedLink"]
    r = requests.get(link) # create HTTP response object
    name = "2000/anger/" + str(i) + "_angry.gif"
    with open(name,'wb') as f:
        f.write(r.content)
    
    link = disgust["content_data"][i]["embedLink"]
    r = requests.get(link) # create HTTP response object
    name = "2000/disgust/" + str(i) + "_disgust.gif"
    with open(name,'wb') as f:
        f.write(r.content)
    
    link = happiness["content_data"][i]["embedLink"]
    r = requests.get(link) # create HTTP response object
    name = "2000/joy/" + str(i) + "_happiness.gif"
    with open(name,'wb') as f:
        f.write(r.content)

    link = surprise["content_data"][i]["embedLink"]
    r = requests.get(link) # create HTTP response object
    name = "2000/surprise/" + str(i) + "_surprise.gif"
    with open(name,'wb') as f:
        f.write(r.content)

#printing the progress
    if(i%100 == 0):
        print(f"{i} files downloaded")

"""
Below code reads files from a directory and writes the same file with modified name
in the destination directory
"""

import os
import cv2

for i, file in enumerate(os.listdir("Emotion6/images/anger/")):
    filename = "Emotion6/images/anger/" + file
    temp_file = cv2.imread(filename)
    filename = "Emotion6/images/negative/" + str(i+1) + "_anger.jpg"
    cv2.imwrite(filename, temp_file)

for i, file in enumerate(os.listdir("Emotion6/images/sadness/")):
    filename = "Emotion6/images/sadness/" + file
    temp_file = cv2.imread(filename)
    filename = "Emotion6/images/negative/" + str(i+1) + "_sadness.jpg"
    cv2.imwrite(filename, temp_file)
    

for i, file in enumerate(os.listdir("Emotion6/images/joy/")):
    filename = "Emotion6/images/joy/" + file
    temp_file = cv2.imread(filename)
    filename = "Emotion6/images/positive/" + str(i+1) + "_joy.jpg"
    cv2.imwrite(filename, temp_file)

for i, file in enumerate(os.listdir("Emotion6/images/surprise/")):
    filename = "Emotion6/images/surprise/" + file
    temp_file = cv2.imread(filename)
    filename = "Emotion6/images/positive/" + str(i+1) + "_surprise.jpg"
    cv2.imwrite(filename, temp_file)
