#importing all necessary libraries
from transformers import BertTokenizerFast as BertTokenizer
from torch.utils.data import Dataset
import torch
import pandas as pd #for loading .csv files

#Inhereting PyTorch Dataset class for handling toxic comments data accordingly.
class Dataset(Dataset):
  def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int = 128):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    self.LABEL_COLUMNS = data.columns.tolist()[2:]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    comment_text = data_row.comment_text
    labels = data_row[self.LABEL_COLUMNS]

    encoding = self.tokenizer.encode_plus(
      comment_text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return dict(
      comment_text=comment_text,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.FloatTensor(labels)
    )