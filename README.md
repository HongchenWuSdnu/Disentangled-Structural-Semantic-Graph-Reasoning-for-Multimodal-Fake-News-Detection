# Disentangled-Structural-Semantic-Graph-Reasoning-for-Multimodal-Fake-News-Detection
Official PyTorch implementation of "Disentangled Structural-Semantic Graph Reasoning for Multimodal Fake News Detection" 
# Disentangled Structural-Semantic Graph Reasoning for Multimodal Fake News Detection

This repository contains the PyTorch implementation of the paper **"Disentangled Structural-Semantic Graph Reasoning for Multimodal Fake News Detection"**.

##  Abstract

Multimodal fake news detection is critical for mitigating the impact of misinformation in online communities. While recent methods have successfully leveraged visual and textual signals, most existing approaches rely on coarse-grained global fusion strategies.

To address this, we propose a **Disentangled Structural-Semantic Graph Reasoning** mechanism. Our method explicitly constructs a dual-graph topology:
1.  **Local Contextual Graph**: Isolates intra-modal structural anomalies.
2.  **Cross-Modal Semantic Graph**: Uses context gating to expose logical conflicts between textual entities and visual regions.

Extensive experiments on **Weibo**, **Twitter**, and **Fakeddit** datasets demonstrate the superiority of our mechanism, achieving state-of-the-art performance by disentangling structural coherence from semantic alignment.

## Project Structure

The repository is organized into three separate folders, each containing the implementation for a specific dataset:

```text
.
├── Fakeddit/           # Implementation for the Fakeddit dataset
├── Twitter/            # Implementation for the Twitter dataset
├── weibo/              # Implementation for the Weibo dataset
└── README.md
```
## Datasets

This project uses public datasets. Please refer to the original repositories for data access and place the data in the corresponding folders:

* **Weibo Dataset**: [Link to EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18)
* **Twitter Dataset**: [Link to CAFE](https://github.com/cyxanna/CAFE)
* **Fakeddit Dataset**: [Link to Fakeddit](https://github.com/entitize/Fakeddit)

## Requirements

* Python 3.8+
* PyTorch >= 1.8
* Transformers (Hugging Face)
* GPU: NVIDIA RTX 4090D (Recommended)

##  Usage

Please enter the corresponding directory to run the experiments.

### 1. Run on Fakeddit Dataset
```bash
cd Fakeddit
# Train and test the model
python fakeddit.py
```
### 2. Run on Weibo Dataset
```bash
cd Weibo
# Train and test the model
python Weibo.py
```
### 3. Run on Twitter Dataset
```bash
cd Twitter
# Train and test the model
python Twitter.py
```
##  Performance

| Dataset | F1-Score | Accuracy |
| :--- | :--- | :--- |
| **Weibo** | 0.9415 | 0.9420 |
| **Twitter** | 0.8914 | 0.8926 |
| **Fakeddit** | 0.8556 | 0.8632 |

##  Citation

If you find this code useful, please cite our work:

```bibtex
@article{yan2026disentangled,
  title={Disentangled Structural-Semantic Graph Reasoning for Multimodal Fake News Detection},
  author={Yan, Yuhan and Wu, Hongchen and Fang, Xiaochang and Du, Ping and Zhang, Huaxiang},
  year={2026},
  note={Manuscript in preparation}
}
```
