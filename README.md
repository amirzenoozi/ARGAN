# ARGANv1   

[Open Sourc]. The improved version of AnimeGANv2.  
[AnimeGANv2](https://tachibanayoshino.github.io/AnimeGANv2/) | Landscape 
photos/videos to anime  
-----  
<br />
**Focus:**  
<table border="1px ridge">
	<tr align="center">
	    <th>Anime style</th>
	    <th>Film</th>  
	    <th>Picture Number</th>  
      <th>Quality</th>
      <th>Download Style Dataset</th>
	</tr >
	<tr align="center">
      <td>Miyazaki Hayao</td>
      <td>The Wind Rises</td>
      <td>1752</td>
      <td>1080p</td>
	    <td rowspan="3"><a href="https://github.com/TachibanaYoshino/AnimeGANv2/releases/tag/1.0">Link</a></td>
	</tr>
	<tr align="center">
	    <td>Makoto Shinkai</td>  
	    <td>Your Name & Weathering with you</td>
      <td>1445</td>
      <td>BD</td>
	</tr>
	<tr align="center">
	    <td>Kon Satoshi</td>
	    <td>Paprika</td>
      <td>1284</td>
      <td>BDRip</td>
	</tr>
</table>  
___ 

____  
## Results
![](https://github.com/amirzenoozi/ARGAN/blob/master/ARGANv1.png)

## Requirements  
You Can Use `requirements` file to install all packages that you need.
  
## Usage  

### 1. Download Pretrained Model    
- [Models](https://github.com/amirzenoozi/models-with-tensorflow/releases)

### 2. Download Train/Val Photo dataset  
  > [Download](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/dataset-1)

### 3. Do edge_smooth  
```bash
python edge_smooth.py --dataset Hayao --img_size 256
```
  
### 4. Calculate the three-channel(BGR) color difference  
```bash
python data_mean.py --dataset Hayao
```
  
### 5. Train  
```bash
python main.py --phase train --dataset Hayao --data_mean [13.1360,-8.6698,-4.4661] --epoch 101 --init_epoch 10
```  
For light version: 
```bash
python main.py --phase train --dataset Hayao --data_mean [13.1360,-8.6698,-4.4661]  --light --epoch 101 --init_epoch 10
```
  
### 6. Extract the weights of the generator  
```bash
python get_generator_ckpt.py --checkpoint_dir  ../checkpoint/ARGANv1_Hayao_lsgan_300_300_1_2_10_1  --style_name Hayao
```

### 7. Inference      
```bash
python test.py --checkpoint_dir  checkpoint/generator_Hayao_weight  --test_dir dataset/test/HR_photo --style_name Hayao/HR_photo
```
  
### 8. Convert video to anime   
```bash
python video2anime.py  --video input.mp4  --checkpoint_dir  checkpoint/generator_Paprika_weight
```  
    
____  
## License  
This repo is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, and scientific publications. Permission is granted to use the ARGANv1 given that you agree to my license terms. Regarding the request for commercial use, please contact us via email to help you obtain the  authorization letter.  

## Citation
If you find our work useful in your research or publications, please consider citing the main paper:
```bash
@INPROCEEDINGS{9738752,
  author={Zenoozi, Amirhossein Douzandeh and Navi, Keivan and Majidi, Babak},
  booktitle={2022 International Conference on Machine Vision and Image Processing (MVIP)}, 
  title={ARGAN: Fast Converging GAN for Animation Style Transfer}, 
  year={2022},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/MVIP53647.2022.9738752}
}
```
And This Repository:
```bash
@software{Douzandeh_Zenoozi_ARGAN_GitHub_Repository_2023,
   author = {Douzandeh Zenoozi, Amirhossein},
   doi = {10.5281/zenodo.8075534},
   month = jun,
   title = {{ARGAN GitHub Repository}},
   url = {https://github.com/amirzenoozi/ARGAN},
   version = {1.0.0},
   year = {2023}
}
```

## Author
Amirhossein Douzandeh Zenoozi
