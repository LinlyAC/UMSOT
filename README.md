# Uncertainty Modeling for Group Re-identification (IJCV 2024)
The implementation for UMSOT, Uncertainty Modeling for Group Re-identification.

## Requirements
### Installation
Please refer to [INSTALL.md](INSTALL.md).

### Datasets
Download the CSG dataset and modify the dataset path, line 26 in [csg.py](./fastreid/data/datasets/CSG.py) (./fastreid/data/datasets/CSG.py):
> self.root = XXX

### Prepare ViT Pre-trained Models
Download the ViT Pre-trained model and modify the path, line 11 in [bagtricks_gvit.yml](./configs/CSG/bagtricks_gvit.yml) (./configs/CSG/bagtricks_gvit.yml):
> PRETRAIN_PATH: XXX

## Training
Single or multiple GPU training is supported. Please refer to scripts folder.

## Acknowledgement
Codebase from [fast-reid](https://github.com/JDAI-CV/fast-reid). So please refer to that repository for more usage.

## Citation
If you find this code useful for your research, please kindly cite the following papers:
'''
@ARTICLE{UMSOT,
  author={Zhang, Quan and Lai, Jianhuang and Feng, Zhanxiang and Xie, Xiaohua},
  title={Uncertainty Modeling for Group Re-Identification}, 
  journal={International Journal of Computer Vision}, 
  year={2024}
}

@article{SOT, 
  title={Uncertainty Modeling with Second-Order Transformer for Group Re-identification}, 
  author={Zhang, Quan and Lai, Jian-Huang and Feng, Zhanxiang and Xie, Xiaohua}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  year={2022}, 
  volume={36}, 
  number={3}, 
  pages={3318-3325}
}
'''

