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
2. Using **download.py** to download the **MovieLens 25M Dataset** and **IMDB(title.crew.tsv.gz)** and unzip these files; or you can download manually.
3. Run the **mergedata.py** to combine the Movielens dataset, TMDB and IMDB together to a movies.csv;
4. Run the **preprocess_for_RQ1.py** and **preprocess_for_RQ2.py** to get the preprocessed files for research_question1 and research_question2;
5. Run the **research_question1.ipynb** and **research_question2.ipynb** to see the data analysis of the movie data.

