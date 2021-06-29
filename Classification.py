# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 23:15:06 2021

@author: hector_ramirez
"""

""" Uncomment and install for the 1st time then comment it """
# conda install -c fastai fastai
# conda install -c fastai fastbook
# conda install pillow



""" Import libraries """
import fastbook
from fastbook import *
from fastai.vision.all import *
import fastai
from skimage import io as io
from PIL import Image
from zipfile import ZipFile
import pandas as pd
import os
import sys
 


""" Functions """
def curved_straight(boolean):
  if boolean == 'False':
      return("Rectos")
  elif boolean == 'True':
    return("Curvos")

def is_curved(x): return x[0] == 'c'



""" Load Deep Learning algorithm """

spermClassifier = Path('C:\\Users\\ramir\\Google Drive\\ML\\Curvos_vs_Rectos\\spermClassifier_18May2021.pkl')

path = Path('C:/Users/ramir/Google Drive/ML/Curvos_vs_Rectos/')
learner = load_learner(path, 'spermClassifier_18May2021.pkl')


""" Load tif image example (one by one) """
img = io.imread('C:\\Users\\ramir\\Google Drive\\ML\\Curvos_vs_Rectos\\43_Ch1.ome.tif') # load 8-bit tif image
img = PILImage.create(img)
img.to_thumb(192)


""" Calculate probabilities """
is_curved,_,probs = learn.predict(img)
print(f"Is this a curved sperm?: {is_curved}.")
print(f"Probability it's a curved sperm: {probs[1].item():.6f}")




""" Analizing in batches """

filename = '/content/drive/MyDrive/Curvos_vs_Rectos/testing_tif.zip' # zip with tif images to analyze

rows_list = []
counter=0
with ZipFile(filename) as archive:
    for entry in archive.infolist():
        with archive.open(entry) as file:
            print(archive.infolist())
            counter+=1
            img = io.imread(file, plugin='tifffile')
            img = PILImage.create(img)
            filename_jpg =  file.name.replace('testing_tif/','') # folder name
            filename = filename_jpg.replace('.tif','') 
            is_curved,_,probs = learn.predict(img) # analyze each image and calculate probabilities
            array = [filename, filename_jpg, entry.file_size, img.width, img.height, curved_straight(is_curved), probs[1].item(), probs[0].item()]
            print(str(round(counter/len(archive.infolist())*100,2))+"%"+" completed ", array)
            rows_list.append(array)
            if curved_straight(is_curved) == 'Curvos':
               img.save('/content/drive/MyDrive/Curvos_vs_Rectos/testing_Curved_vs_Straights/curved/'+ str(round(probs[1].item()*100,1)) + '%_' + filename_jpg)
            else: 
               img.save('/content/drive/MyDrive/Curvos_vs_Rectos/testing_Curved_vs_Straights/straight/'+ str(round(probs[1].item()*100,1)) + '%_' + filename_jpg)


DF = pd.DataFrame(rows_list, columns=['image name','image', 'size (bytes)', 'width (px)', 'height (px)', 'Classification', 'Prob Curved', 'Prob Straight'])
DF.to_csv('/content/drive/MyDrive/Curvos_vs_Rectos/testing_Curved_vs_Straights/test_CSV.csv')   


