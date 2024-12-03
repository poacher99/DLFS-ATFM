# Decoupled Facial Aging Image Generation with Enhanced Age-Transform Coding



## Preparation

Please follow [this github](https://github.com/royorel/Lifespan_Age_Transformation_Synthesis) to prepare the environments and dataset.

## Training and Testing 

The pre-trained generative model can be downloaded from [here](https://drive.google.com/file/d/1vbkm4r_PK__bKhmhg61WOd4CX5aeXUhc/view?usp=sharing)

### Linux
training (please modify `--dataroot`, `--name`):
```
sh train_distan.sh
```
testing (please modify `--dataroot`, `--name`, `--which_epoch`, and `--checkpoing_dir`):
```
sh test_distan.sh
```

### Windows
training
```
python train_distan.py --gpu_ids 0 --dataroot ./datasets/males --name males_model --batchSize 4 --encoder_type distan --verbose --display_id 0
```

testing
```
python test_distan.py --verbose --dataroot ./datasets/males --name males_model  --display_id 0 --encoder_type distan 
```

## Acknowledgements

This repository is a modified version of the code from [DLFS](https://github.com/SenHe/DLFS), which is licensed under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The modifications made in this repository are also shared under the same license.

The test dataset we used, FFHQ Aging, can be found [here](https://github.com/royorel/FFHQ-Aging-Dataset/tree/master). This dataset also follows the CC BY-NC-SA 4.0 License.

