# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 06:10:01 2018

@author: Akrem
"""

import numpy as np
import os, gdal

#in_path = 'C:/Users/Akrem/Desktop/ECP/3 eme Annee/DIL/santiago/'
#input_filename = 'Santiago_ortho_rec_RGB.tif'

#in_path = 'C:/DIL_Project/CNN1-master/raw/train/'
#input_filename = 'austin1_mask.tif'

in_path = 'C:/DIL_Project/CNN1-master/raw/train/'
input_filename = 'austin4_mask.tif'

#out_path = 'C:/DIL_Project/CNN1-master/raw/train/'
#output_filename = 'tile_'
#out_path = 'C:/DIL_Project/CNN1-master/raw/train/austin1_mask/'
#output_filename = 'austin1_'

out_path = 'C:/DIL_Project/CNN1-master/raw/train/austin1/'
output_filename = 'austin4_'


tile_size_x = 400
tile_size_y = 400
 
ds = gdal.Open(in_path + input_filename)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + "_mask" + ".tif"
        os.system(com_string)
        print(i,j)