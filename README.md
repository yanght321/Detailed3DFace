# Detailed Riggable 3D Face Prediction
**[[Arxiv]](https://arxiv.org/abs/2003.13989)**
<img src="https://github.com/yanght321/Detailed3DFace/teaser/1.png" >

## Requirements
- The project is tested on ubuntu 18.04 with python 3.6, PyTorch 1.6.0.  
- Blilinear model from [FaceScape dataset](https://facescape.nju.edu.cn/Page_Data/) is required.  You have to download the bilinear model after submitting an application.

## Usage
1. We recomend using [Anaconda](https://www.anaconda.com/download/#linux) to create a new enviroment:
```
conda create -n 3dface python=3.6
conda activate 3dface
```

Install dependecies:
```
pip install -r requirements.txt
```
Refer to the [guide](https://pytorch.org/get-started/locally/) to install the PyTorch 1.6.0.

2. Download the Blilinear model from FaceScape dataset. After getting access to the dataset, download 'facescape_bilinear_model_v1_3.zip' from the download page. Then put 'core_847_50_52.npy' and 'factors_id_847_50_52.npy' into './predef' folder.

3. Download the pretrained model from [Google Drive](https://drive.google.com/file/d/18j8bnj5IHP0u2jNuIrWh7dvQkfagBxsM/view?usp=sharing), unzip the file and place 'dpmap_rig_net_G.pth' and 'dpmap_single_net_G.pth' in './checkpoints' folder.

4. Run the demo code.
```
python main.py --name dpmap_single --input ./demo/input --output ./demo/output --gpu_ids 0
```
The results will be saved in the folder './demo/output/'. Displacement maps can be loaded and visualized in MAYA, ZBrush, 3D MAX, etc. The results in our paper is rendered in ZBrush and the scale is set to 0.014.

You can set '--name dpmap_rig' to predict displacement maps corresponding to the 20 key expressions. The input image should be in nearly neutral expression for this option.

Note that the base model fitting results may be different from that in our paper, because our landmark detector cannot be released due to the licenses. So we used the publicly available dlib detector as 
substituent and simplified the fitting algorithm.


## Citation

Please cite the following paper if this project helps your research:
```
@InProceedings{yang2020facescape,
  author = {Yang, Haotian and Zhu, Hao and Wang, Yanru and Huang, Mingkai and Shen, Qiu and Yang, Ruigang and Cao, Xun},
  title = {FaceScape: A Large-Scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020},
  page = {601--610}}
```
