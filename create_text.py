import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 
from PIL import Image,ImageDraw,ImageFont
import argparse
from os import listdir
import os.path as osp
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--font_path',default = 'D:/newfont')
parser.add_argument('--des_path',default = 'D:/vietnamese_character_1k')
args = parser.parse_args()

SIZE = 1000
cnt = 0

def load_font(path):
	ret = []
	for x in listdir(path):
		ret.append(osp.join(path,x))
	return ret

def Extend(x):
	while len(x)!=5:
		x = '0' + x
	return x

if __name__ == '__main__':
	fonts = load_font(args.font_path)
	if not os.path.exists(osp.join(args.des_path,'image')):
		os.makedirs(osp.join(args.des_path,'image'))
	if not os.path.exists(osp.join(args.des_path,'label')):
		os.makedirs(osp.join(args.des_path,'label'))

	char_list = ['a','ă','â','b','c','d','đ','e','ê','g','h','i','k','l','m','n',
	'o','ô','ơ','p','q','r','s','t','u','ư','v','x','y']
	for char in char_list: 
		for i in range(SIZE):
			image = np.zeros((305,400,3),np.uint8)
			image = image + 255
			img = Image.fromarray(image, 'RGB')

			font = ImageFont.truetype(random.choice(fonts), np.random.randint(150,200))
			color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
			pos = (np.random.randint(0,200),np.random.randint(0,150))

			draw = ImageDraw.Draw(img)
			draw.text(pos,char,color,font=font)
			path = osp.join(args.des_path,'image' + Extend(str(cnt)) + '.jpg')
			print(path)
			img.save(path)
			cnt += 1
		