'''
Author: Shuailin Chen
Created Date: 2021-05-18
Last Modified: 2021-05-19
	content: 1) adapt from Song Hui's matlab code, only the pixel-level 
				distance, thresholding, evaluation parts are available in this script till now.
			 2) otsu's method are not suitable for this unsupervised method
'''
import os
import os.path as osp

from scipy import ndimage
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import mylib.polSAR_utils as psr
from mylib import mathlib

import PolSAR_distance_metric as pdm
from GHT import GHT


def boxcar_smooth(file, kernel_size):
	''' boxcar smoothing 

	Args:
		file (ndarray): file to be filterd
		kernel_size (int): filter kernel size

	Returns:
		filtered image
	'''

	ff_r = ndimage.uniform_filter(file.real, (0, kernel_size, kernel_size), mode='mirror')
	ff_i = ndimage.uniform_filter(file.imag, (0, kernel_size, kernel_size), mode='mirror')
	return ff_r+1j*ff_i


def imadjust(im, quantile=0.01):
	''' imadjust func like Matlab's 
	
	Args:
		im (ndarray): image 
		quantile (float): in [0, 0.5]. Default: 0.01

	Returns:
		adjusted image, normalized to in [0, 1]
	'''

	assert quantile>0 and quantile<0.5
	upper = np.quantile(im, 1-quantile)
	lower = np.quantile(im, quantile)

	im = np.clip(im, lower, upper)
	return (im-im.min()) / (im.max()-im.min())


def unsupervised_CD(fa_path, fb_path, save_path, gt_path=None, distance_type='srw', is_print=False):
	''' Unsupervised change detection

	Args:
		fa_path (str): 时相A数据所在目录，仅接受 C3 数据
    	fb_path (str): 时相B数据所在目录，仅接受 C3 数据
    	save_path (str): 保存结果的路径
    	gt_path (str): groundtruth path, if assigned, evaluation will be 
			performed. Default: None
		distance_type (str): type of distance metric. Default: 'srw'
		is_print (bool): whether to print infos. Default: False
	
	Returns: None if gt_path not specified, otherwise Confusion matrix whose 
		i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class
	'''

	# read c3 data
	c31 = psr.read_c3(fa_path)
	c32 = psr.read_c3(fb_path)
	if is_print:
		print(f'shape: {c31.shape}')

	# boxcar smooth to ensure non-negative definete
	c31 = boxcar_smooth(c31, 3)
	c32 = boxcar_smooth(c32, 3)
	p31 = psr.rgb_by_c3(c31)
	p32 = psr.rgb_by_c3(c32)
	cv2.imwrite(osp.join(save_path, 'boxcar_1.png'), (p31*255).astype(np.uint8))
	cv2.imwrite(osp.join(save_path, 'boxcar_2.png'), (p32*255).astype(np.uint8))

	# pixel distance
	dist = pdm.distance_by_c3(c31, c32, distance_type)
	dist = (dist-dist.min()) / (dist.max()-dist.min())
	cv2.imwrite(osp.join(save_path, 'dist.png'), (imadjust(dist)*255).astype(np.uint8))
	# plt.hist(dist.flatten())
	# plt.savefig(osp.join(save_path, 'hist_ori.png'))
	# plt.clf()

	# thresholding
	dist[dist<mathlib.eps] = mathlib.eps
	dist = np.log(dist)
	dist = (psr.min_max_map(dist)*255).astype(np.uint8)

	# plt.hist(dist.flatten(), range(256))
	# plt.savefig(osp.join(save_path, 'hist_aft.png'))
	# plt.clf()

	# thres = 0.95
	# result = (dist>thres).astype(np.uint8)
	# cv2.imwrite(osp.join(save_path, 'result.png'), 255*result)
	# thres = GHT(np.histogram(dist, 256)[0], None, 2**29.5, 2**3.125, 2**22.25, 2**(-3.25))[0]
	# result = (dist>thres).astype(np.uint8)
	# cv2.imwrite(osp.join(save_path, 'result.png'), 255*result)

	# result = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# cv2.imwrite(osp.join(save_path, 'result.png'), result[1])

	thres = GHT(np.histogram(dist, 256)[0])[0]
	result = (dist>thres).astype(np.uint8)
	cv2.imwrite(osp.join(save_path, 'result.png'), 255*result)

	# confusion matrix
	gt = cv2.imread(gt_path)
	if gt.ndim>=3:
		gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
		gt = (gt>128).astype(np.uint8)
		return confusion_matrix(gt.flatten(), result.flatten())


if __name__ == '__main__':
    fa = r'data/2009_SUB_SUB/C3'
    fb = r'data/2010_SUB_SUB/C3'
    gt = r'data/suzhou_gt4.bmp'
    save_path = r'tmp'
	
    confusion_matrix = unsupervised_CD(fa, fb, save_path, gt, is_print=True, 
										distance_type='Bartlett')

    print(f'confusion_matrix:\n{confusion_matrix}')

    print('done')