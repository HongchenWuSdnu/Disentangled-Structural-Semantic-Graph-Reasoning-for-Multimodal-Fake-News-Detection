# Disentangled-Structural-Semantic-Graph-Reasoning-for-Multimodal-Fake-News-Detection
Official PyTorch implementation of "Disentangled Structural-Semantic Graph Reasoning for Multimodal Fake News Detection" (Submitted to Expert Systems with Applications
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
