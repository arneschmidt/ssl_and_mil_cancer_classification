import argparse
import xml.etree.ElementTree as ET
import math 
from PIL import Image, ImageDraw 
from PIL import ImagePath  
import openslide as sld

import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def main(args):
    camelyon_dir = args.camelyon_dir
    annotation_dir = args.annotation_dir
    save_dir = args.save_dir
    image_dir = args.image_dir

    if not os.path.isdir(camelyon_dir+'training/'+save_dir):
        os.mkdir(camelyon_dir+'training/'+save_dir)
        print('save dir created')

    for xml_file in glob.glob(camelyon_dir+'training/'+annotation_dir+'/*.xml')[77:]:
        print (xml_file)
        tree=ET.parse(xml_file)
        root=tree.getroot()
        annotations=root[0]

        file=xml_file.split("/")[-1]
        name=file.split(".")[0]

        image_slide = sld.open_slide(camelyon_dir+'training/'+image_dir+'/'+name+'.tif')
        size=image_slide.dimensions

        img = Image.new("L", size, 0)
        for annotation in annotations:
            coordinates= annotation[0]
            coord_list=[]
            for coordinate in coordinates:
                thistuple=(float(coordinate.attrib["X"]),float(coordinate.attrib["Y"]))
                coord_list.append(thistuple)
                #tenemos el poligono completo y lo pintamos
            image = ImagePath.Path(coord_list).getbbox()
            #size = list(map(int, map(math.ceil, image[2:])))
            #print(annotation.attrib['Name'],annotation.attrib['PartOfGroup'])
            #print(annotation.attrib['Color'])
            #img = Image.new("RGB", size, "#ffffff")
            img1 = ImageDraw.Draw(img)
            #img1.polygon(coord_list, fill ="#000000", outline =None)
            if annotation.attrib['PartOfGroup']=='_0':
                img1.polygon(coord_list, fill =255, outline =None)  #FF0000
                #print("skip")
            elif annotation.attrib['PartOfGroup']=='_1':
                img1.polygon(coord_list, fill =255, outline =None)  #00FF00
            elif annotation.attrib['PartOfGroup']=='_2':
                img1.polygon(coord_list, fill =0, outline =None) #0000FF
            else:
                print('ERROR: Group not found')

        #img.save("mask_v2.jpg")
        img.save(camelyon_dir+'training/'+save_dir+'/'+name+'_annotation_mask.png')
        #img.save("evaluation_mask_prueba.png")
        print(name+" mask saved")
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camelyon_dir", "-i", type=str, help='path to base directory of Camelyon16')
    parser.add_argument("--save_dir", "-i", type=str, help="output directory")
    parser.add_argument("--annotation_dir", "-i", type=str, help='directory of annotation files', default='lesion_annotation')
    parser.add_argument("--image_dir", "-i", type=str, help='directory of the WSI images', default='tumor')

    args = parser.parse_args()
    print('Arguments:')
    print(args)
    main(args)