## Introduction:

This repo is worked as a computing artifact for **Tianjie Zhang**'s comprehensive exam. The related works have been published in IEEE-T-ITS.


## Data:

The **images** folder contains all the data utilized in this part. In the images floder, it contains four sub-floders: 


| Subfolders      | Info                                           |   Training | 
| :------------:   | :-------------:                                  |:--------:|
| cracks          |  [DeepCrack](https://github.com/yhlleo/DeepCrack)(contains 587 images and their annotation)| 300 from [DeepCrack](https://github.com/yhlleo/DeepCrack)|
| cracks-tradition| traditional augmentation   |300 from [DeepCrack](https://github.com/yhlleo/DeepCrack) + 300 from traditional augmentation |
| cracks-DCGAN     |   [DCGAN](https://github.com/soumith/dcgan.torch)     |300 from [DeepCrack](https://github.com/yhlleo/DeepCrack) + 300  from [DCGAN](https://github.com/soumith/dcgan.torch) |
| cracks-APCGAN|    APC-GAN    | 300  from [DeepCrack](https://github.com/yhlleo/DeepCrack) + 300 from APCGAN    |      

The **testing data** are all from the DeepCrack which contains **287** crack images.

## Models:

Models list (used in the paper): 

- AttuNet
- AttuNet-min
- [U-Net](https://github.com/milesial/Pytorch-UNet)
- [FCN](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.fcn_resnet50.html#torchvision.models.segmentation.fcn_resnet50)
- [LRASPP](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html#torchvision.models.segmentation.lraspp_mobilenet_v3_large)
- [DeepLabv3](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html#torchvision.models.segmentation.deeplabv3_resnet50)


## Run Order:
### First, setup the environment on your PC.
 Create the environment in your terminal: 
```python 
conda env create -f environment.yml 
```
You can also update an environment to make sure it meets the current requirements in a file:
```python 
conda env update -f environment.yml
```
### Second, augment the images.

There are three ways to augment your training dataset in the **augmentation** folder:

- Random crop (traditional augmentation method)

run **random_crop.py** : images produced will be stored in augmentation/results/random_crop


- [DCGAN](https://github.com/soumith/dcgan.torch)

run **DCGAN.py** : generated real-like images will be stored in augmentation/results/DCGAN

- APCGAN

run **APCGAN.py** : generated real-like images will be stored in augmentation/results/APCGAN


After get the generated images, annotate these images manually and then put them into the training_images floders to be used for deep-learning training.

### Third, training different deep learning methods on the datasets.

1. run the **train.py**: 

1.1  choose and replace a model which you want to run.

```python 
'''
 Choose a deep learning method:

    1. net = AttU_Net(img_ch=1, output_ch=1)

    2. net = AttU_Net_min(img_ch=1, output_ch=1)
    
    3. net = UNet(n_channels=1, n_classes=1)

    4. net = fcn_resnet50(num_classes=1)

    5.  net = deeplabv3_resnet50(num_classes= 1)

    6. net = lraspp_mobilenet_v3_large(num_classes=1)
'''
```

1.2 choose the training dataset
```python 
  data_path = "./images/cracks" # todo: your training datasets
  '''
  four choices: 
  "./images/cracks"
  "./images/cracks_tradition"
  "./images/cracks_DCGAN"
  "./images/cracks_APCGAN"
  '''
```
1.3 you will get a **best_model.pth** after you run the **train.py**.

2. Run the **test.py**

do some modifications when you run the test.py:
```python 
 net = AttU_Net(img_ch=1, output_ch=1) # todo: change the model refering to your trained model
 ...

 net.load_state_dict(torch.load('best_model.pth', map_location=device)) # todo
 # todo: make you load your best_model.pth
```
2.1 you will get the evaluation results of your model in the **results** folder.


PS: when taining model **fcn_resnet50** ,**deeplabv3_resnet50** , **lraspp_mobilenet_v3_large**, remember to modify the last layer, making it consistent with the classes in this work.

---


### By doing these above, you finished a model setup and evaluation. 
### Then, you can do the same steps on all the models.
### Finally, compare the evalution metrics among models.





