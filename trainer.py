#importing the necessary libraries and modules
from transformers import BertModel, BertTokenizer
from data_loader_and_preprocessor import dataLoader, dataInterpreter
from pytorch_lightning.callbacks import ModelCheckpoint
from data_module import dataModule
from train_valid_tagger import trainValidTagger
import pytorch_lightning as pl


# defining basic parameter for the model training
N_EPOCHS = 2
BATCH_SIZE = 32
MAX_TOKEN_COUNT = 512
train_data, valid_data, LABEL_COLUMNS = dataLoader()
BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)

def main():
    #initializing the dataInterpreter class to preprocess the data
    dataInterpreter()
    #initializing the ToxicCommentDataModule by passing the relevant arguments
    data_module = dataModule(
    train_data, 
    valid_data,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_token_len=MAX_TOKEN_COUNT
    )

    #Calculating the necessary parameters for the model training
    steps_per_epoch=len(train_data) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = total_training_steps // 5
    warmup_steps, total_training_steps

    #instantiating the train_valid_tagger Class with necessary parameters to make it prepare for training
    model = trainValidTagger(
    n_classes=len(LABEL_COLUMNS),
    n_warmup_steps=warmup_steps,
    n_training_steps=total_training_steps
    )

    #Defining a callback that save the best model during training based on the validation loss.
    checkpoint_callback = ModelCheckpoint(
    dirpath="Bert_model/checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
    )

    #Configuring the pytorch-lightning trainer object with basic parameters necessory for our model training
    #this can also be done with pytorch trainer however that consume a lot of resource and therefore oftnely crashed the colab
    trainer = pl.Trainer(
    max_epochs=N_EPOCHS,
    callbacks=[checkpoint_callback],
    devices='auto',
    accelerator = 'auto',
    enable_progress_bar=True
    )

    #initializing the training process by passing the model (an object of the ToxicCommentTagger Class) to the to the trainer.fit function of pytroch-lightning
    trainer.fit(model, data_module)
if __name__=='__main__':
    main()