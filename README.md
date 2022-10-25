APCGAN-AttuNet


## Data:

The image folder contains all the data utilized in this part. In the images floder, there contains three sub-floders: 


| subfolders      | info                                           |      
| ------------    | -------------                                  |
| cracks          |      300 original images from [DeepCrack](https://github.com/yhlleo/DeepCrack)        |       
| cracks-gan      |                original + gan                  |  
| cracks-tradition|         original + traditional augmentation    |         

## Models:
All the models are in the model file.

AttuNet




## Run Order:
1. Create the environment in your terminal: 
```python 
conda env create -f environment.yml 
```
You can also update an environment to make sure it meets the current requirements in a file:
```python 
conda env update -f environment.yml
```
2. Run the APCGAN

3. Run the train.py file.
