# VAAL in PyTorch

**Update**: For any questions, please email me at samarth.sinha@mail.utoronto.ca, since I often don't get GitHub emails. 

Original Pytorch implementation of "Variational Adversarial Active Learning" (ICCV 2019 **Oral**). [Link to the paper.](https://arxiv.org/abs/1904.00370)

Samarth Sinha*, Sayna Ebrahimi*, Trevor Darrell, Internation Conference on Computer Vision (ICCV 2019)

*First two authors contributed equally*


## Abstract 
Active learning aims to develop label-efficient algorithms by sampling the most representative queries to be labeled by an oracle. We describe a pool-based semi-supervised active learning algorithm that implicitly learns this sampling mechanism in an adversarial manner. Unlike conventional active learning algorithms, our approach is \textit{task agnostic}, i.e., it does not depend on the performance of the task for which we are trying to acquire labeled data. Our method learns a latent space using a variational autoencoder (VAE) and an adversarial network trained to discriminate between unlabeled and labeled data. The mini-max game between the VAE and the adversarial network is played such that while the VAE tries to trick the adversarial network into predicting that all data points are from the labeled pool, the adversarial network learns how to discriminate between dissimilarities in the latent space. We extensively evaluate our method on various image classification and semantic segmentation benchmark datasets and establish a new state of the art on *CIFAR10/100*, *Caltech-256*, *ImageNet*, *Cityscapes*, and *BDD100K*. Our results demonstrate that our adversarial approach learns an effective low dimensional latent space in large-scale settings and provides for a computationally efficient sampling method.

  
## Citation
If using this code, parts of it, or developments from it, please cite our paper:
```
@article{sinha2019variational,
  title={Variational Adversarial Active Learning},
  author={Sinha, Samarth and Ebrahimi, Sayna and Darrell, Trevor},
  journal={arXiv preprint arXiv:1904.00370},
  year={2019}
}
```

### Prerequisites:
- Linux or macOS
- Python 3.5/3.6
- CPU compatible but NVIDIA GPU + CUDA CuDNN is highly recommended.

### Installation
The required Python3 packages can be installed using 
```
pip3 install -r requirements.txt
```

### Experiments and Visualization
The code can simply be run using 
```
python3 main.py
```
When using the model with different datasets or different variants, the main hyperparameters to tune are
```
--adversary_param --beta --num_vae_steps and --num_adv_steps
```

The results will be saved in `results/accuracies.log`. 

**We have provided the code and data required to replicate all plots in paper [here](https://github.com/sinhasam/vaal/blob/master/plots/plots.ipynb).**


### License
The entire codebase is under BSD 2-Clause License.


## Contact
If there are any questions or concerns feel free to send a message at samarth.sinha@mail.utoronto.ca
