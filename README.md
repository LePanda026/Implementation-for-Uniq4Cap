# About
This is the code for Uniq4Cap,which proposed in “UniQ4Cap Unified Query Representation from Pre-aligned Video and Audio for Multimodal Video Captioning”


**Authors**: Junshuo Zhang<sub>#, Guangyu Wang<sub>#, Jing Liu<sub>#, **Ming Gao<sub>{#*}**, Zhiqiao Wu<sub>#, Jiafu Tang<sub>#, Bowei Chen<sub>#, Weiguo Fan<sub>#. Dongbei University of Finance and Economics, all the authors contributed equally to this work. Corresponding author: Ming Gao.
## 💡 Highlights
### ✨Superior performance, but relative low demand in computation
UniQ4Cap is a **lightweight** multimodal video captioning method that **distills useful information** for multimodal video captioning from **jointed pre-aligned video and audio features**.

The following first figure shows the architecture and training pipeline of UniQ4Cap. UniQ4Cap can be easily extended to single modlity captioning task, downstream classification and retrieval tasks.
* **First stage**: The video and audio encoders are aligned in an **unified semantic space**. We use contrastive learning techniques to map the pre-trained encoders to a unified semantic space. Considering **different modalities have different granularity**, we use both Noise Contrastive Estimation (NCE) and Multiple Instance Learning NCE (MIL-NCE) to optimize the aligment process.
* **Second stage**: A lightweight multimodal video captioning model is trained, which includes two backbone networks: a query transformer and a text transformer. The model extracts the **most text-related multimodal features** through query vectors, cross attention, and shared self-attention weights.
<img src="https://github.com/LePanda026/Implementation-for-Uniq4Cap/blob/main/model.png" />

### 📺 A multimodal dataset containing video, audio and multimodal textual description
We contribute a new dataset that includes both **video and audio, with annotations describing both modalities**. The dataset comprises a training set of **100K** video, and the test set includes **35K**.
The second figure shows our proposed dataset, which includes three modalities: video, audio, and language. Due to policy constraints, we are unable to directly distribute the videos. Instead, we offer YouTube IDs that allow you to download the videos on your own. You can access all textual sources and YouTube IDs for download from [Google Disk](https://drive.google.com/file/d/160P8r5Hc9IcZ5wsuCpIocfbaZMh9U7Gq/) or [Baidu Disk](https://pan.baidu.com/s/1HBmcnHW2HZYX57p_QnvUhg?pwd=9teq).

<img src="https://github.com/LePanda026/Implementation-for-Uniq4Cap/blob/main/dataset.png" />

## 🔧 Quick Start
* Prepare virtual environment
```bash
conda create -n UniQ4Cap python=3.8.18
conda activate UniQ4Cap
```

* Clone our repository
```bash
git clone https://github.com/LePanda026/Implementation-for-Uniq4Cap.git
cd Implementation-for-Uniq4Cap
cd UniQ4Cap
```

* Install required packages:
```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116  
pip install -r requirements1.txt  
pip install -r requirements.txt  
pip install torch==2.1.2  
pip install torchvision==0.16.2  
pip install torchaudio==2.1.2  
```

* Download the dataset and modify the following parameters in UniQ4Cap/script:
```bash
root_dir: 'your root dir to your raw data and annotations'
data_path: 'raw data dir'
ann_path: 'annotations.json'
```

* Download the model and modify the following parameters in UniQ4Cap/script:  
All checkpoints can be availible here:   
Bert: 🤗[Huggfacing](https://huggingface.co/google-bert); Pre-aligned Encoder: [Baidu Disk](https://pan.baidu.com/s/1CxF7U0GTo8VMvLgFdT94LQ?pwd=bdss); Query Transformer: [Baidu Disk](https://pan.baidu.com/s/1s23pI-lVXUeIks_-ptUx9Q?pwd=sac1).
```bash
  vit_config: "/mnt/ha/bd2/LAVIS/mm_llm/configs/video_config.json"
  aud_config: "/mnt/ha/bd2/LAVIS/mm_llm/configs/audio_config.json"
  vit_pretrained: ""
  aud_pretrained: ""
  bert_pretrained: ""
  qformer_pretrained: ""
```

* Start Training
```bash
python -m torch.distributed.launch --nproc_per_node 8 train_prealign.py
```

## 💖 Acknowledgement
* [OpenCLIP](https://github.com/mlfoundations/open_clip) An open source pretraining framework.
* [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) An open source Video-Text retrieval framework.
* [LAVIS](https://github.com/salesforce/LAVIS) A One-stop Library for Language-Vision Intelligence.
