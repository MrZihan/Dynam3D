### Dynam3D: Dynamic Layered 3D Tokens Empower VLM for Vision-and-Language Navigation

#### Zihan Wang, Seungjun Lee, Gim Hee Lee

> Vision-and-Language Navigation (VLN) is a core task where embodied agents leverage their spatial mobility to navigate in 3D environments toward designated destinations based on natural language instructions. Recently, video-language large models (Video-VLMs) with strong generalization capabilities and rich commonsense knowledge have shown remarkable performance when applied to VLN tasks. However, these models still encounter the following challenges when applied to real-world 3D navigation: 1) Insufficient understanding of 3D geometry and spatial semantics; 2) Limited capacity for large-scale exploration and long-term environmental memory; 3) Poor adaptability to dynamic and changing environments.To address these limitations, we propose Dynam3D, a dynamic layered 3D representation model that leverages language-aligned, generalizable, and hierarchical 3D representations as visual input to train 3D-VLM in navigation action prediction. Given posed RGB-D images, our Dynam3D projects 2D CLIP features into 3D space and constructs multi-level 3D patch-instance-zone representations for 3D geometric and semantic understanding with a dynamic and layer-wise update strategy. Our Dynam3D is capable of online encoding and localization of 3D instances, and dynamically updates them in changing environments to provide large-scale exploration and long-term memory capabilities for navigation. By leveraging large-scale 3D-language pretraining and task-specific adaptation, our Dynam3D sets new state-of-the-art performance on VLN benchmarks including R2R-CE, REVERIE-CE and NavRAG-CE under monocular settings. Furthermore, experiments for pre-exploration, lifelong memory, and real-world robot validate the effectiveness of practical deployment.

[Huggingface](https://huggingface.co/datasets/MrZihanWang/Dynam3D)
[Navigation Data](https://huggingface.co/datasets/MrZihanWang/Dynam3D/tree/main/data/datasets)
[arXiv](https://arxiv.org/abs/2505.11383)
## TODOs

* [x] Release the pre-training code of Dynam3D.
* [x] Release the pre-training checkpoints of Dynam3D.
* [x] Release the pre-training datasets of Dynam3D.
* [x] Release the code of vision-language navigation.
* [ ] Release the checkpoints of vision-language navigation.
* [x] Release the training datasets of vision-language navigation.

### Requirements

1. Create a Conda environment. We developed this project with Python 3.8.
   ```
   conda env create -f environment.yaml
   conda activate dynam3d
   ```

2. Install `habitat simulator [v0.1.7](https://github.com/facebookresearch/habitat-lab/releases/tag/v0.1.7) and `habitat-lab [v0.1.7](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7)` : follow instructions from [ETPNav](https://github.com/MarSaKi/ETPNav) or [VLN-CE](https://github.com/jacobkrantz/VLN-CE).

3. Install `torch_kdtree` for K-nearest feature search from [torch_kdtree](https://github.com/thomgrand/torch_kdtree).
   
   ```
   git clone https://github.com/thomgrand/torch_kdtree
   cd torch_kdtree
   git submodule init
   git submodule update
   pip3 install .
   ```

4. Install `tinycudann` for faster multi-layer perceptrons (MLPs) from [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
   
   ```
   pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
   ```

5. Download the preprocessed data and checkpoints from [Hugging Face](https://huggingface.co/datasets/MrZihanWang/Dynam3D).
   
6. (Optional) Download other Pre-training data.
    Download RGB-D images of [ARKitScenes](https://github.com/apple/ARKitScenes)
    Download RGB-D images of [Structured3D](https://github.com/bertjiazheng/Structured3D)

### (Optional) Pre-train the Dynam3D Representation Model

```
cd Dynam3D_Pretrain
bash run_3dff/3dff.bash train 2341
python3 convert_ckpt.py # Convert the pre-trained checkpoint for downstream tasks, i.e., dynam3d.pth
```
### Train the Dynam3D-VLN Model
Please check the navigation training and validation data [Navigation Data](https://huggingface.co/datasets/MrZihanWang/Dynam3D/tree/main/data/datasets), and make the necessary modifications to [task.py])(Dynam3D_VLN/habitat_extensions/task.py) and [r2r_vlnce.yaml](Dynam3D_VLN/scripts/r2r_vlnce.yaml).
```
cd Dynam3D_VLN
bash scripts/main.bash train 2344 # training
bash scripts/main.bash eval 2344 # evaluation
bash scripts/main.bash inter 2344 # inference
```
## Citation

```bibtex
@inproceedings{wang2025dynam3d,
  title={Dynam3D: Dynamic Layered 3D Tokens Empower VLM for Vision-and-Language Navigation},
  author={Wang, Zihan and Lee, Seungjun and Lee, Gim Hee},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgments

Our code is based on [llava-phi-3-mini-hf](https://huggingface.co/xtuner/llava-phi-3-mini-hf), [g3D-LF](https://github.com/MrZihan/g3D-LF) and [ETPNav](https://github.com/MarSaKi/ETPNav). Thanks for their great works!
