#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 2018

@author: tmtrcreations
"""

# Import the needed libraries
import scipy
from PIL import Image

# Loop through the Morty sprites and check sizes and remove alpha channel
for image_ind in range(1, 199):
    
    if((image_ind != 115) & (image_ind != 116) & (image_ind != 146)):
        
        # Load in the images
        img = scipy.misc.imread("../../Datasets/Sample_Sprites/Morty" + "%03d" % image_ind + ".png", mode='RGBA')
        img = Image.fromarray(img)
        
        # Check the sizes
        if((img.height != 125) & (img.width != 148)):
            print("Wrong size!")
            
        # Create the new image
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        background.save("../../Datasets/Sample_Sprites/Morty" + "%03d" % image_ind + ".png")