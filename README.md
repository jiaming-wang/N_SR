# N_SR
 
There are some implements of SR methods with Pytorch.  <br>
* RCAN in rcan.py ([Image Super-Resolution Using Very Deep Residual Channel Attention Networks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf))
* DBPN in dbpn.py ([Deep Back-Projection Networks For Super-Resolution](https://arxiv.org/abs/1904.05677))
* RDN in rdn.py ([Residual Dense Network for Image SR](https://arxiv.org/pdf/1802.08797v2.pdf))
* EDSR in edsr.py ([Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921))
* SRResNet in srresnet.py ([Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802))

## Dependencies
* Python 3.5.3
* Pytorch 1.1.0
* tensorboardX

### Getting started
* Train: `python main.py`. More details in `option.py`</br>
* Test: `python eval.py`.

## Experiments on FEI face dateset
Image-based algoritnms.
<table>
	<tr>
		<td><center> </center></td>
		<td><center>Bicubic</center></td>
		<td><center>EDSR_original</center></td>
		<td><center>EDSR+b16k64</center></td>
		<td><center>EDSR+b32k256</center></td>
		<td><center>SRResNet</center></td>
		<td><center>RDN</center></td>
		<td><center>DBPN</center></td>
		<td><center>RCAN</center></td>
	</tr>
	<tr>
		<td>
			<center>PSNR</center>
		</td>
		<td>
			<center>36.38</center>
		</td>
		<td>
			<center>39.81</center>
		</td>
		<td>
			<center>39.85</center>
		</td>
		<td>
			<center>40.05</center>
		</td>
		<td>
			<center>40.04</center>
		</td>
		<td>
			<center>39.90</center>
		</td>
		<td>
			<center>40.14</center>
		</td>
		<td>
			<center>40.03</center>
		</td>
	</tr>
</table>
Patch-based algoritnms.
<table>
	<tr>
		<td><center> </center></td>
		<td><center>Bicubic</center></td>
		<td><center>VDSR_original</center></td>
	</tr>
	<tr>
		<td>
			<center>PSNR</center>
		</td>
		<td>
			<center>36.38</center>
		</td>
		<td>
			<center>39.54</center>
		</td>
	</tr>
</table>
`tensorboard --logdir log`</br>

The train/test loss and PSNR curves for each experiment are shown below:</br>
<p align="center"><img src="./log.png" align="center" width=300 height=500/></p>
