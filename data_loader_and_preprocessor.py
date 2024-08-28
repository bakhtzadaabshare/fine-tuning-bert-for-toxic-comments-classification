#importing all necessary libraries
import pandas as pd #for loading .csv files
from sklearn.model_selection import train_test_split

def dataLoader():
  #This is toxic comments classification dataset avaliable on Kaggle.
  df = pd.read_csv("../toxic_comments.csv")
  #splitting the whole data into training and test part in order to train and validate the model performance
  train_df, val_df = train_test_split(df, test_size=0.05)
  #Taking the last six classes for classification which toxic, severe_toxic,	obscene,	threat,	insult, and	identity_hate
  LABEL_COLUMNS = df.columns.tolist()[2:]
  return train_df, val_df, LABEL_COLUMNS

#Displaying basis statistics about the dataset
def dataInterpreter():
  train_df, val_df, LABEL_COLUMNS = dataLoader()
  print(f'The shape of the train set: {train_df.shape}\n')
  print(f'The shape of the validation set: {val_df.shape}\n')
  #Displaying the number of toxic and clean comments for the purpose to analyze the dataset
  train_toxic = train_df[train_df[LABEL_COLUMNS].sum(axis=1) > 0]
  train_clean = train_df[train_df[LABEL_COLUMNS].sum(axis=1) == 0]
  print(f"This is the number of toxic comments found in the train set: {train_toxic.shape}\n")
  print(f"This is the number of clean commments found in the train set: {train_clean.shape}\n")