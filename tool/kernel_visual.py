import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys, torch
import os, math, random
from dataset import *

gen_kernel = Gaussin_Kernel(kernel_size=21, blur_type='aniso_gaussian', sig=2.6, sig_min=0.2, sig_max=4.0,
                lambda_1=0.5, lambda_2=4.0, theta=45, lambda_min=0.2, lambda_max=4.0)
blur = BatchBlur(kernel_size=21)
b_kernels = gen_kernel(1, random)

b_kernels = b_kernels.squeeze()
plt.imshow(np.array(b_kernels))
plt.axis('off')
plt.show()

