# Pedestrian Phone Usage detection and tracking using AlphaPose estimation and pose tracking model 


## Installation
##### 1. Create a conda virtual environment.

conda config --append channels conda-forge
 
conda create -n trackPhoneUsage python=3.6 pandas scikit-learn  matplotlib opencv spyder notebook
 
conda activate trackPhoneUsage

##### 2. Install PyTorch
conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch

##### 3. Get Track_Phone_Usage project repo
git clone https://github.com/aivoyagers/Track_Phone_Usage.git
cd Track_Phone_Usage

##### 4. install
`export PATH=/usr/local/cuda/bin/:$PATH`

`export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH`

`pip install cython`

- sudo apt-get install libyaml-dev
`pip install pyyaml`    

`python setup.py build develop`

##### 5. For Windows user, if you meet error with PyYaml, you can download and install it manually from here: https://pyyaml.org/wiki/PyYAML. If your OS platform is Windows, make sure that Windows C++ build tool like visual studio 15+ or visual c++ 2015+ is installed for training.

##### 6. Download data and pretrained/offline trained Models from google drive : https://drive.google.com/drive/folders/1ZyzRj-G6FolQ-qhJzOx1YkBptYtaWjeS?usp=sharing .
- Copy all the folders in the same structure from the google drive to project (git cloned) base folder. 
- Note: Track_Phone_Usage/data folder is required only for training and not required for testing. 

' copy [build, data, demo, detector, pretrained_models, samples, yolo] to Track_Phone_Usage (git cloned project base folder)`

##### 7. To train on labelled video datasets in 'data' folder :
` 'python track_phone_usage.py --video data --train'  `

##### 8. To predict or test :
` 'python track_phone_usage --video /path/to/video_file.mp4'  `

<div align="center">
    <img src="demo/track_phone_usage_demo.gif", width="400">
</div>


## Citation
Please cite these papers in your publications if it helps your research:

    @inproceedings{fang2017rmpe,
      title={{RMPE}: Regional Multi-person Pose Estimation},
      author={Fang, Hao-Shu and Xie, Shuqin and Tai, Yu-Wing and Lu, Cewu},
      booktitle={ICCV},
      year={2017}
    }

    @article{li2018crowdpose,
      title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
      author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
      journal={arXiv preprint arXiv:1812.00324},
      year={2018}
    }

    @inproceedings{xiu2018poseflow,
      author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
      title = {{Pose Flow}: Efficient Online Pose Tracking},
      booktitle={BMVC},
      year = {2018}
    }


## License
AlphaPose is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail at mvig.alphapose[at]gmail[dot]com and cc lucewu[[at]sjtu[dot]edu[dot]cn. We will send the detail agreement to you.
