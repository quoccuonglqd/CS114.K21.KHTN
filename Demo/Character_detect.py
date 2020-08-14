import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import cv2 as cv

from craft import CRAFT
from collections import OrderedDict

def sorting_key(x):
	return x[0]

class Character_detect(object):
	def __init__(self):
		self.net = CRAFT()
		self.net.load_state_dict(self.copyStateDict(torch.load("weight/craft_mlt_25k.pth", map_location='cpu')))
		self.net.eval()

	def test_net(self,net, image, text_threshold, link_threshold, low_text, poly, refine_net=None):
		img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv.INTER_LINEAR, mag_ratio=1.5)
		ratio_h = ratio_w = 1 / target_ratio
		x = imgproc.normalizeMeanVariance(img_resized)
		x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
		x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

		with torch.no_grad():
			y, feature = net(x)

		# make score and link map
		score_text = y[0,:,:,0].cpu().data.numpy()
		score_link = y[0,:,:,1].cpu().data.numpy()

		# Post-processing
		boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

		# coordinate adjustment
		boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
		polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
		for k in range(len(polys)):
			if polys[k] is None: polys[k] = boxes[k]

		# render results (optional)
		render_img = score_text.copy()
		render_img = np.hstack((render_img, score_link))
		ret_score_text = imgproc.cvt2HeatmapImg(render_img)

		return boxes, polys, ret_score_text

	def detect(self,path):
		image = imgproc.loadImage(path)
		refine_net = None
		bboxes, polys, score_text = self.test_net(self.net, image, 0.7, 999999, 0.5, False, refine_net)
		bbox = []
		for i, box in enumerate(polys):
			poly = np.array(box).astype(np.int32).reshape((-1))
			bbox.append([poly[0]-3,poly[1]-5,poly[2],poly[5]+5])
		file_utils.saveResult(path, image[:,:,::-1], polys, dirname="Detect_result/")
		bbox.sort(key=sorting_key)
		return bbox

	def copyStateDict(self,state_dict):
		if list(state_dict.keys())[0].startswith("module"):
			start_idx = 1
		else:
			start_idx = 0
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			name = ".".join(k.split(".")[start_idx:])
			new_state_dict[name] = v
		return new_state_dict