# <p align=center>`DFormer for RGBD Semantic Segmentation`</p>
This repository is forked from an official implementation of the following papers:

> DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation<br/>
> [Bowen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=zh-CN&oi=sra),
> [Xuying Zhang](https://scholar.google.com/citations?hl=zh-CN&user=huWpVyEAAAAJ),
> [Zhongyu Li](https://scholar.google.com/citations?user=g6WHXrgAAAAJ&hl=zh-CN),
> [Li Liu](https://scholar.google.com/citations?hl=zh-CN&user=9cMQrVsAAAAJ),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=zh-CN&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=zh-CN) <br/>
> ICLR 2024. 
>[Paper Link](https://arxiv.org/abs/2309.09668) |
>[Homepage](https://yinbow.github.io/Projects/DFormer/index.html) |
>[公众号解读(集智书童)](https://mp.weixin.qq.com/s/lLFejycBr8o7JNoirRDmjQ) |
>[DFormer-SOD](https://github.com/VCIP-RGBD/DFormer-SOD) |
>[Jittor-Version(国产框架)](https://github.com/VCIP-RGBD/DFormer-Jittor) |


> DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation<br/>
> [Bo-Wen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=zh-CN&oi=sra),
> [Jiao-Long Cao](https://github.com/caojiaolong),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=zh-CN&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=zh-CN)<br/>
> CVPR 2025. 
> [Paper Link](https://arxiv.org/abs/2504.04701) |
> [中文版](https://mftp.mmcheng.net/Papers/25CVPR_RGBDSeg-CN.pdf) |
> [直播回放](https://www.bilibili.com/video/BV1hGNozuEe4?t=4.2) |
> [PPT](https://pan.baidu.com/s/1HjmiVBYZSnBGcPDJgfCeoA?pwd=ti6p) |
> [Geometry prior demo](https://huggingface.co/spaces/bbynku/DFormerv2) |
> [Jittor-Version(国产框架)](https://github.com/VCIP-RGBD/DFormer-Jittor) |



## Reference
You may want to cite:
```
@inproceedings{yin2024dformer,
  title={DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation},
  author={Yin, Bowen and Zhang, Xuying and Li, Zhong-Yu and Liu, Li and Cheng, Ming-Ming and Hou, Qibin},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{yin2025dformerv2,
  title={DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation},
  author={Yin, Bo-Wen and Cao, Jiao-Long and Cheng, Ming-Ming and Hou, Qibin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={19345--19355},
  year={2025}
}
```

### License

Code in this repo is for non-commercial use only.




How to use this method:



<br><br>
## **1) Environment setup**



```bash

conda create -n dformer python=3.10 -y

conda activate dformer



# CUDA 11.8

conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia



pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html



pip install tqdm opencv-python scipy tensorboardX tabulate easydict ftfy regex

```





 <br><br>
## **2) Download data set and weights**



You can use your own data set or download one from the internet. In the official implementation repository in README.md you can find links to the data set developers used for testing. 

In this graduation thesis I used a data set downloaded from: https://dataverse.csuc.cat/dataset.xhtml?persistentId=doi:10.34810/data916. 

Pretrained weights can also be downloaded from the links provided in the official implementation repository. 




<br><br>
## **3) Prepare data set**



In the file "data_set_preperation.py" you can find methods that were used to prepare the data set used in this work. 

Organize the checkpoints and dataset folder in the following structure:


```shell
<checkpoints>
|-- <pretrained>
    |-- <DFormer_Large.pth.tar>
    |-- <DFormer_Base.pth.tar>
    |-- <DFormer_Small.pth.tar>
    |-- <DFormer_Tiny.pth.tar>
    |-- <DFormerv2_Large_pretrained.pth>
    |-- <DFormerv2_Base_pretrained.pth>
    |-- <DFormerv2_Small_pretrained.pth>
<datasets>
|-- <DatasetName1>
    |-- <RGB>
        |-- <name1>.<ImageFormat>
        |-- <name2>.<ImageFormat>
        ...
    |-- <Depth>
        |-- <name1>.<DepthFormat>
        |-- <name2>.<DepthFormat>
        ...
    |-- <Label>
        |-- <name1>.<LabelFormat>
        |-- <name2>.<LabelFormat>
        ...
    |-- train.txt
    |-- test.txt
    |-- val.txt
|-- <DatasetName2>
|-- ...
```

</code></pre>
</details>


<br><br>
## **4) Train**



You can change the `local_config' files in the script to choose the model for training. 
 Settings used in this paper are defined in the file Diplomski-rad/local_configs/Fuji/DFormer_Small.py.  You can adjust parameters if you need to.
You run the script for training by executing the following command in the terminal: 

```

bash train.sh

```



After training, the checkpoints will be saved in the path `checkpoints/XXX', where the XXX depends on the training config. 




<br><br>
## **5) Test**



You can change the `local_config' files and checkpoint path in the script to choose the model for testing. 
 You run the script for testing by executing the following command in the terminal: 

```

bash eval.sh

``` 









