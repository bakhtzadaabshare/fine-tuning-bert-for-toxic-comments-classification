from train_valid_tagger import trainValidTagger
from data_loader_and_preprocessor import dataLoader
from trainer import trainer, tokenizer
import torch

LABEL_COLUMNS = dataLoader()[2]

#loading the best model from the checkpionts
trained_model = trainValidTagger.load_from_checkpoint(
  trainer.checkpoint_callback.best_model_path,
  n_classes=len(LABEL_COLUMNS)
)
trained_model.eval()
trained_model.freeze()
trained_model = trained_model.to('cpu')

#inferencing the model performance on our given comments
toxic_comment = "Hello man, what the hell you are doing. You're such an idiot"
clean_comment = 'I know that this product is not quite good but I still like it.'

def comment_classification(comments):
  encoding = tokenizer.encode_plus(
      comments,
      add_special_tokens=True,
      max_length=512,
      return_token_type_ids=False,
      padding="max_length",
      return_attention_mask=True,
      return_tensors='pt',
    )
  _, test_prediction = trained_model(encoding["input_ids"], encoding["attention_mask"])
  test_prediction = test_prediction.flatten().cpu().numpy()
  return test_prediction

#calling the function for toxic_comment
print("This is the classification of toxic comment\n")
for label, prediction in zip(LABEL_COLUMNS, comment_classification(toxic_comment)):
  print(f"{label}: {prediction}")

#calling the function for clean_comment
print("--------------*******************************************--------------------")
print("This is the classification of clean comment\n")
for label, prediction in zip(LABEL_COLUMNS, comment_classification(clean_comment)):
  print(f"{label}: {prediction}")