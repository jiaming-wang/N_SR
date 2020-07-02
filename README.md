# N_SR
 
There are some implements of image super-resolution methods with Pytorch.  <br>
* SAN in san.py ([Second-order Attention Network for Single Image Super-Resolution](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf)) (2020)
* RCAN in rcan.py ([Image Super-Resolution Using Very Deep Residual Channel Attention Networks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf)) (2019)
* DBPN in dbpn.py ([Deep Back-Projection Networks For Super-Resolution](https://arxiv.org/abs/1904.05677)) (2018)
* RDN in rdn.py ([Residual Dense Network for Image SR](https://arxiv.org/pdf/1802.08797v2.pdf)) (2018)
* EDSR in edsr.py ([Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)) (2017)
* SRResNet in srresnet.py ([Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)) (2016)
* SRCNN in srcnn.py ([Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)) (2014)

## Dependencies
* Python 3.5.3
* Pytorch 1.1.0
* tensorboardX

### Getting started
Image-based algoritnms.
* Train: `python main.py`. </br>
* Test: `python test.py`. More details in `option.py`.</br>

Patch-based algoritnms.
* Image to patch:`python image_to_patch.py`. </br>
* Train: `python main.py`. </br>
* Test: `python test.py`. More details in `option.py`.</br>

## Experiments on FEI face dateset (without augmentation and pre-train)
Image-based algoritnms.
|Algorithms|Bicubic|EDSR_original|EDSR+b16k64|EDSR+b32k256|SRResNet without BN|RDN|DBPN|RCAN|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|PSNR | 36.38 | 39.81 | 39.85 | 40.05 | 40.04 | 39.90 | 40.14 | 40.03 |

Patch-based algoritnms.
|Algorithms|Bicubic|SRCNN_original|SRCNN|VDSR_original|
|:---:|:---:|:---:|:---:|:---:|
|PSNR | 36.38 | 38.58 | 38.61 |39.54 |

## License
This project is released under the Apache 2.0 license.
