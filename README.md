APCGAN-AttuNet


## Data:

The image folder contains all the data utilized in this part. In the images floder, there contains three sub-floders: 


| subfolders      | info                                           |      
| ------------    | -------------                                  |
| cracks          |      300 original images from DeepCrack        |       
| cracks-gan      |                original + gan                  |  
| cracks-tradition|         original + traditional augmentation    |         


## Run Order:
1. Create the environment with conda env create: 
```python 
conda env create -f environment.yml 
```
You can also update an environment to make sure it meets the current requirements in a file:
```python 
conda env update -f environment.yml
```
