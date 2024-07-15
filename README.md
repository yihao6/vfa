# Vector Field Attention
This is the official Pytorch implementation of the deformable registration algorithm: "Vector Field Attention" we introduced in our paper: "Vector Field Attention for Deformable Image Registration."

## Prerequisites
VFA currently support 3D images in Nifti format. At least a moving image and a fixed image needs to be provided.
Additionally, VFA allows the following optional inputs:
- Fixed/moving image mask
- Fixed/moving image label map
Masks can be used to remove the background in the input images or remove the background region in loss during training.
Label maps enables Dice loss during training.

Inhomogeneity correction recommended for structural MR images.
We used [Tustison, Nicholas J., et al. "N4ITK: improved N3 bias correction."](https://ieeexplore.ieee.org/document/5445030)

## Installation
### Docker / Singularity container
The easist way to run our algorithm is through Docker or Singularity containers.

For Docker, you can download the Docker image using
```bash
    docker pull registry.gitlab.com/iacl/vfa:vX.Y.Z
```

For singularity, you can use
```bash
    singularity pull --docker-login docker://registry.gitlab.com/iacl/vfa:vX.Y.Z
```

Singularity image can also be directly downloaded [**here**](https://iacl.ece.jhu.edu/~yihao/vfa/vfa_vX.Y.Z.sif).

### Installation from source code
1. Clone this repository:
```bash
    git clone https://github.com/yihao6/vfa.git
```
2. Navigate to the directory:
```bash
    cd vfa
```
3. Install dependencies
```bash
    pip install .
```

### Pretrained weights
Pretrained weights of VFA can be downloaded [**here**](https://iacl.ece.jhu.edu/~yihao/vfa/vfa_vX.Y.Z.pth).

## Usage
If you use the Docker container, see:
```bash
    docker run -it registry.gitlab.com/iacl/vfa:vX.Y.Z vfa-run --help
```

If you the Singularity container, see:
```bash
    singularity exec ./vfa_vX.Y.Z.sif vfa-run --help
```

If you installed from source code, see:
```
    vfa-run --help
```

### Training
To start training, use the following command:
```bash
vfa-run train [-h] [--output_dir OUTPUT_DIR] --identifier IDENTIFIER --train_data_configs TRAIN_DATA_CONFIGS [--gpu GPU] [--checkpoint CHECKPOINT] [--params PARAMS]
                     [--eval_data_configs EVAL_DATA_CONFIGS] [--cudnn]

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        Output directory (default: ./vfa)
  --identifier IDENTIFIER
                        A string that identify the current run (required)
  --train_data_configs TRAIN_DATA_CONFIGS
                        Path to data_configs.json for training data information (required)
  --gpu GPU             GPU ID. Default: 0
  --checkpoint CHECKPOINT
                        Path to pretrained models.
                            During training, continue from this checkpoint.
                            During evaluation, evaluate this checkpoint.
  --params PARAMS       Path to params.json for hyper-parameters.
                            If a checkpoint is provided, defaults to params.json in the checkpoint folder.
  --eval_data_configs EVAL_DATA_CONFIGS
                        Path to data_configs.json for evaluation data information
  --cudnn               Enable CUDNN for potential speedup
```

### Evaluation
To evaluate the model, use the following command:
```bash
vfa-run evaluate [-h] [--save_results SAVE_RESULTS] [--f_img F_IMG] [--m_img M_IMG] [--f_input F_INPUT] [--m_input M_INPUT] [--f_mask F_MASK] [--m_mask M_MASK] [--f_seg F_SEG] [--m_seg M_SEG]
                        [--prefix PREFIX] [--gpu GPU] [--checkpoint CHECKPOINT] [--params PARAMS] [--eval_data_configs EVAL_DATA_CONFIGS] [--cudnn]

optional arguments:
  -h, --help            show this help message and exit
  --save_results SAVE_RESULTS
                        Specify level of evaluation results to save:
                            0: no results saved
                            1: save minimal outputs
                            2: save all inputs and outputs
  --gpu GPU             GPU ID. Default: 0
  --checkpoint CHECKPOINT
                        Path to pretrained models. During training, continue from this checkpoint. During evaluation, evaluate this checkpoint.
  --params PARAMS       Path to params.json for hyper-parameters. If a checkpoint is provided, defaults to params.json in the checkpoint folder.
  --eval_data_configs EVAL_DATA_CONFIGS
                        Path to data_configs.json for evaluation data information
  --cudnn               Enable CUDNN for potential speedup
```

If eval_data_configs not provided, you can also provide the following path in command line
```bash
  --f_img F_IMG         Path to fixed image
  --m_img M_IMG         Path to moving image
  --f_input F_INPUT     Path to fixed input image
  --m_input M_INPUT     Path to moving input image
  --f_mask F_MASK       Path to fixed mask
  --m_mask M_MASK       Path to moving mask
  --f_seg F_SEG         Path to fixed label map
  --m_seg M_SEG         Path to moving label map
  --prefix PREFIX       Prefix for saved results
```

The central component to the training and evaluation process is preparing the data_config.json files. It contains information to correctly load the data. Examples can be found in vfa/data_configs/.

Detailed information regarding each in a valid json file is provided below:
1. "loader": data loader to be used to load the data. It should be a class name defined in vfa/datasets/
    We have implemented commonly used ones for most cases.
    - "Pairwise": pairwise registration. In this case, you need to provide a "pairs" list in the json file that contains pairs of images to be registered.
        This should be the most commonly used data loader.
    - "Inter": inter subject registration. In this case, you need to provide a "f_subjects" list and a "m_subjects" list in the json file.
        A fixed image will be randomly chosen from the "f_subjects" set
        and a moving image will be randomly chosen from the "m_subjects" set. If "m_subjects" is not defined
        in the json file, both fixed and moving image will be randomly chosen from the "f_subjects" set.
        This data laoder can also be used for registration between subjects and atlas images
        by specifying the subject images in one set and atlas images in the other set.
        In generally, this data loader should only be used during training.
    - "Intra": intra subject registration. In this case, you need to provide a "subjects" list.
        Each element inside the "subjects" list should also be a list that contains individual images of the same subject.
        First, a specific subject is randomly chosen. Then fixed and moving image will be randomly chosen from the selected subject.
        In generally, this data loader should only be used during training.
2. "shape": the dimension of the inputs. Our code use this information to determine whether to use 2D or 3D network.
    The first dimension should be the number of channels, followed by the spatial dimensions.
    The actual number of the spatial dimensions are not used by VFA.
3. "transform": the preprocessing steps for the input images.
    If your data has been preprocessed, both fixed and moving images has the same spatial
    dimensions that can be divided by 2^5, you only need
    ```bash
    "transform":[                                                                                     
        {"class_name":"Nifti2Array"},                                                                 
        {"class_name":"DatatypeConversion"},                                                          
        {"class_name":"ToTensor"}                  
    ],
    ```
    for more general applications, our default is
    ```bash
    "transform" = [                                                                  
        {"class_name":"Reorient", "orientation":'RAS'},                                       
        {"class_name":"Resample", "target_res":[1.0, 1.0, 1.0]},                              
        {"class_name":"Nifti2Array"},                                                                                                                                                               
        {"class_name":"AdjustShape", "target_shape":[192, 224, 192]},                                                                                                                               
        {"class_name":"DatatypeConversion"},                                                  
        {"class_name":"ToTensor"},                                                            
    ]
    ``` 
4. "labels": labels used for computing Dice loss (optional). You can include a subset of all available labels
    in the label map input to encourage the registration to focus on certain structure.
5. "m_subjects" / "f_subjects" / "subjects" / "pairs": depends on the data loader, you need to specifies these lists
    Each image is a dictionary that contains the following information:
    ```bash
    {
        "id":38,
        "img":"/path/to/image.nii.gz",
        "mask":"/path/to/image_mask.nii.gz",
        "seg":"/path/to/image_label_map.nii.gz",
        "prefix":"/path/to/results_prefix_"
    },
    ```
    Only "img" is required.

We recommend using the examples provided in vfa/data_configs/ and modifies those examples to avoid error.

### Training outputs
Checkpoint files will be saved to output_dir/checkpoints/identifier/
Tensorboard files will be saved to output_dir/runs/

output_dir and identifier are specified as command line argument during training

### Evaluation outputs
Registration results are saved based on prefix setting in data_config.json
When save_results is set to 1, VFA saves:
1. The warped image: "prefix_m_img.nii.gz"
2. The intersection of the warped and fixed mask: "prefix_mask.nii.gz"
3. The transformation grid: "prefix_grid.nii.gz"
4. (When the moving segmentation is provided) The warped label map: "prefix_w_seg.nii.gz"

When save_results is set to 2, VFA saves additional images:
5. The fixed image: "prefix_f_img.nii.gz"
6. The moving image: "prefix_m_img.nii.gz"
7. The warped mask: "prefix_w_mask.nii.gz"
8. The fixed label map: "prefix_f_seg.nii.gz"
9. The moving label map: "prefix_m_seg.nii.gz"
10. The grid line representation of the transformation: "prefix_grid_lines.nii.gz"
11. Displacement magnitude image: "prefix_disp_magnitude.nii.gz"
We export those additional images to be useful for debugging purpose, however, they will slow down the processing

### Citation
If you use this code, please cite our papers.

and our previous conference paper:
```
@inproceedings{liu2022coordinate,
  title={Coordinate translator for learning deformable medical image registration},
  author={Liu, Yihao and Zuo, Lianrui and Han, Shuo and Xue, Yuan and Prince, Jerry L and Carass, Aaron},
  booktitle={International Workshop on Multiscale Multimodal Medical Imaging},
  pages={98--109},
  year={2022},
  organization={Springer}
}
```
