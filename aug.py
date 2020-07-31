from imgaug import augmenters as iaa
from os import listdir
import random
import cv2

# seq = iaa.Sequential([iaa.GaussianBlur(sigma=(0.0, 3.0)),
# 		                  iaa.FastSnowyLandscape(
#     lightness_threshold=(100, 255),
#     lightness_multiplier=(1.0, 4.0)
# )])
seq = iaa.imgcorruptlike.Spatter(severity=2)
file_ls = listdir('D:\\vietnamese_character_1k')
aug_ls = random.choices(file_ls,k=2500)

for image in aug_ls:
	img = cv2.imread('D:\\vietnamese_character_1k\\' + image)
	img_aug = seq(image=img)
	cv2.imwrite('D:\\vietnamese_character_1k\\' + image, img_aug)
