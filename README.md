# VeGaS-4D

### Dependencies
 
Install other packages including PyTorch with CUDA (this repo has been tested with CUDA 11.8), [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), and PyTorch3D.
`cuda-toolkit` is required for building `tiny-cuda-nn`.

For CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt200/download.html
pip install --upgrade pip setuptools
```

If you got any issues from the above installation, see [Installation documentation](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md) from nerfstudio for more.

### Installation

```bash
git clone https://github.com/JasiekKaczmarczyk/VeGaS-4D.git
cd VeGaS-4D; pip install -e .
cd libs/diff-gaussian-rasterization-confidence; pip install .
cd ../knn; pip install .
cd ../knn_ops_3_fwd_bwd_mask; pip install .
```
If you have successfully reached here, you are ready to run the code! 

## Training
### Train model
For training synthetic scenes from D-NeRF Dataset such as `bouncingballs`, run 
```
ns-train splatfacto --data $data_root$/dnerf/bouncingballs
```

## 4. Rendering and Evaluation

### Render testing images 
Run the following command to render the images.  
```bash
ns-render dataset --load_config $path_to_your_experiment$/config.yml --output-path $path_to_your_experiment$ --split test
```
If you followed all the previous steps, `$path_to_your_experiment$` should look
something like `outputs/bouncing_balls/splatfacto/2024-XX-XX_XXXXXX`.
### Calculating testing PSNR
```bash
python scripts/metrics.py $path_to_your_experiment$/test
```

## Citation

The codebase is based on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and [4D-Rotor-Gaussian-Splatting](https://github.com/weify627/4D-Rotor-Gaussians).

```
@misc{smolakdyżewska2024vegasvideogaussiansplatting,
      title={VeGaS: Video Gaussian Splatting}, 
      author={Weronika Smolak-Dyżewska and Dawid Malarz and Kornel Howil and Jan Kaczmarczyk and Marcin Mazur and Przemysław Spurek},
      year={2024},
      eprint={2411.11024},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.11024}, 
}
```

```
@inproceedings{duan:2024:4drotorgs,
   author = "Yuanxing Duan and Fangyin Wei and Qiyu Dai and Yuhang He and Wenzheng Chen and Baoquan Chen",
   title = "4D-Rotor Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes",
   booktitle = "Proc. SIGGRAPH",
   year = "2024",
   month = July
}
```

