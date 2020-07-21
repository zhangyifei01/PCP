# Progressive Cluster Purification for Unsupervised Feature Learning

Comparison and Pipeline.

![PCP-Fig1](https://img-blog.csdnimg.cn/20200721001531308.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg3MjU3OA==,size_16,color_FFFFFF,t_70)

![PCP-Fig2](https://img-blog.csdnimg.cn/20200721003129969.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg3MjU3OA==,size_16,color_FFFFFF,t_70)

This is a PyTorch implementation of the [PCP paper](https://arxiv.org/abs/2007.02577):

```
@Article{zhang2020pcp,
  author  = {Yifei Zhang and Chang Liu and Yu Zhou and Wei Wang and Weiping Wang and Qixiang Ye},
  title   = {Progressive Cluster Purification for Unsupervised Feature Learning},
  journal = {arXiv preprint arXiv:2007.02577},
  year    = {2020},
}
```

#### Abstract：
In unsupervised feature learning, sample specificity based methods ignore the inter-class information, which deteriorates the discriminative capability of representation models. Clustering based methods are error-prone to explore the complete class boundary information due to the inevitable class inconsistent samples in each cluster. In this work, we propose a novel clustering based method, which, by iteratively excluding class inconsistent samples during progressive cluster formation, alleviates the impact of noise samples in a simple-yet-effective manner. Our approach, referred to as Progressive Cluster Purification (PCP), implements progressive clustering by gradually reducing the number of clusters during training, while the sizes of clusters continuously expand consistently with the growth of model representation capability. With a well-designed cluster purification mechanism, it further purifies clusters by filtering noise samples which facilitate the subsequent feature learning by utilizing the refined clusters as pseudo-labels. Experiments on commonly used benchmarks  demonstrate that the proposed PCP improves baseline method with significant margins. 


#### Reference：
We thank the open source code [CVPR18: Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://github.com/zhirongw/lemniscate.pytorch) and [CVPR19: Unsupervised Embedding Learning via Invariant and Spreading Instance Feature](https://github.com/mangye16/Unsupervised_Embedding_Learning).


#### Contact:
Yifei Zhang for mails zhangyifei0115@iie.ac.cn .
