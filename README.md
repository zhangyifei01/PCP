# Progressive Cluster Purification for Unsupervised Feature Learning

Pipeline of PCP.

![pcp_fig1](https://github.com/zhangyifei0115/PCP/tree/master/pic/PCP-Fig1.png)

![pcp_fig2](https://github.com/zhangyifei0115/PCP/tree/master/pic/PCP-Fig2.png)

This is a PyTorch implementation of the [PCP paper](https://arxiv.org/abs/2007.02577):

```
@Article{zhang2020pcp,
  author  = {Yifei Zhang and Chang Liu and Yu Zhou and Wei Wang and Weiping Wang and Qixiang Ye},
  title   = {Progressive Cluster Purification for Unsupervised Feature Learning},
  journal = {arXiv preprint arXiv:2007.02577},
  year    = {2020},
}
```

#### Usage：

There are two folders **PCP** and **pcp-imagenet100**, each folder contains the entire project and can be run separately.
  
Just run command 

```bash
sh main.sh
```

by modifyinng the parameters of file **main.sh**.

#### Abstract：
In unsupervised feature learning, sample specificity based methods ignore the inter-class information, which deteriorates the discriminative capability of representation models. Clustering based methods are error-prone to explore the complete class boundary information due to the inevitable class inconsistent samples in each cluster. In this work, we propose a novel clustering based method, which, by iteratively excluding class inconsistent samples during progressive cluster formation, alleviates the impact of noise samples in a simple-yet-effective manner. Our approach, referred to as Progressive Cluster Purification (PCP), implements progressive clustering by gradually reducing the number of clusters during training, while the sizes of clusters continuously expand consistently with the growth of model representation capability. With a well-designed cluster purification mechanism, it further purifies clusters by filtering noise samples which facilitate the subsequent feature learning by utilizing the refined clusters as pseudo-labels. Experiments on commonly used benchmarks  demonstrate that the proposed PCP improves baseline method with significant margins. 

#### More about Dataset:

Imagenet-100 includes classes:

n02869837, n01749939, n02488291, n02107142, n13037406, n02091831, n04517823, 
n04589890, n03062245, n01773797, n01735189, n07831146, n07753275, n03085013, 
n04485082, n02105505, n01983481, n02788148, n03530642, n04435653, n02086910, 
n02859443, n13040303, n03594734, n02085620, n02099849, n01558993, n04493381, 
n02109047, n04111531, n02877765, n04429376, n02009229, n01978455, n02106550, 
n01820546, n01692333, n07714571, n02974003, n02114855, n03785016, n03764736, 
n03775546, n02087046, n07836838, n04099969, n04592741, n03891251, n02701002, 
n03379051, n02259212, n07715103, n03947888, n04026417, n02326432, n03637318, 
n01980166, n02113799, n02086240, n03903868, n02483362, n04127249, n02089973, 
n03017168, n02093428, n02804414, n02396427, n04418357, n02172182, n01729322, 
n02113978, n03787032, n02089867, n02119022, n03777754, n04238763, n02231487, 
n03032252, n02138441, n02104029, n03837869, n03494278, n04136333, n03794056, 
n03492542, n02018207, n04067472, n03930630, n03584829, n02123045, n04229816, 
n02100583, n03642806, n04336792, n03259280, n02116738, n02108089, n03424325, 
n01855672, n02090622.


#### Reference：
We thank the open source code [CVPR18: Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://github.com/zhirongw/lemniscate.pytorch), [ECCV2018: Deep Clustering for Unsupervised Learning of Visual Features](https://github.com/facebookresearch/deepcluster), [CVPR19: Unsupervised Embedding Learning via Invariant and Spreading Instance Feature](https://github.com/mangye16/Unsupervised_Embedding_Learning) and [ICML19: Unsupervised Deep Learning by Neighbourhood Discovery](https://github.com/Raymond-sci/AND).


#### Contact:
Yifei Zhang for mails zhangyifei0115@iie.ac.cn .
