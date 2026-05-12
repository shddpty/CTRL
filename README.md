# CTRL

Code for paper: [Learning Compact Semantic Information and Reliable Pseudo-labels for Incomplete Multi-View Multi-Label Classification](https://ieeexplore.ieee.org/document/11397552)
Pascal07 data is prepared for a demo, you can download data from [here](https://drive.google.com/drive/folders/1cixudQjQ1I1pvRFy0Srsl2UTgvzv-o2k?usp=drive_link).
You can run 'python main_pascal07.py' for a demo!

## Abstract
Multi-view data encompasses various data types, including multi-feature, multi-sequence, and multi-modal data. Multi-view multi-label classification aims to leverage the rich semantic information contained in multiple views to achieve enhanced multi-label classification performance. In practical applications, the absence of views and labels poses a significant challenge to multi-view multi-label classification tasks. Premised on the assumption that shared semantic information across multiple views is sufficient to support the downstream task, we propose CTRL, a novel incomplete multi-view multi-label classification framework to address the multi-view learning challenge on the data with partially missing views and missing labels in this paper. The core mechanism of CTRL lies in learning a high-purity, lowredundancy condensed representation that adequately captures the essential information of the original data. Specifically, we design a new objective loss to enhance the semantic information of shared cross-view within the joint representation learning process while simultaneously suppressing intra-view redundant information that is irrelevant to the downstream task. This enables CTRL to extract task-relevant representations even when views are incomplete. Furthermore, we employ the Beta Evidential Neural Network to model the label distribution. This network is then integrated with Dempster-Shafer theory, enabling our model to perform label-level classification uncertainty estimation. This also allows us to use the estimated uncertainty and belief mass to create high-reliability pseudo-labels, resulting in further gains in model performance. Experimental results on multiple benchmark datasets demonstrate the superior performance of our proposed model in terms of accuracy, robustness, and reliability.

## Overview of CTRL
<img width="2714" height="1441" alt="image" src="https://github.com/user-attachments/assets/37fd6dae-64e5-4c20-b354-fb23358374e2" />



## Environment
Please run the following command in the shell, as specified in `requirements.txt`:
```bash
conda create -n CTRL
conda activate CTRL
pip install -r requirements.txt
```

## Citation
If you find this work useful, please consider citing it:
```
@article{liu2026learning,
  title={Learning compact semantic information and reliable pseudo-labels for incomplete multi-view multi-label classification},
  author={Liu, Yadong and Liu, Chengliang and Wen, Jie and Shen, Li and Zhang, Bob and Xu, Yong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  publisher={IEEE}
}
```
