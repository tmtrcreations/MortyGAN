#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 2018

@author: tmtrcreations
"""

# Import the needed modules
import requests
from bs4 import BeautifulSoup

# Request the Spriters Resource page
r = requests.get("https://www.spriters-resource.com/mobile/pocketmortys/")

# Convert the data to soup
data = r.text
soup = BeautifulSoup(data, "lxml")

# Pull the sample sprites
for link in soup.find_all('img'):

    # Pull the alt tag
    alt_tag = link.get('alt')
    
    # Check for the Mortys
    if(('Mort' in alt_tag) & ('Rick' not in alt_tag) & ('#' in alt_tag)):
       
       # Print the alt tag
       print(alt_tag)
       
       # Get the source of the image
       img_tag = 'https://www.spriters-resource.com' + link.get('src')
       
       # Get the image from the site
       r2 = requests.get(img_tag)
       
       # Set the image name for saving
       file_name = '../../Datasets/Sample_Sprites/Morty' + alt_tag[1:4] + '.png'
       
       # Save the image
       with open(file_name, "wb") as f:
           f.write(r2.content)
    
    