# Fine Tuning BERT-BASE for Toxic Comments Classification (MultiLabel Classification)
In this project, we used the BERT-base-case version for the toxic comment classification task, which is a multilabel task. 
Transformer library from Hugging Face ![image](https://github.com/user-attachments/assets/fd565975-ba72-47de-8bf2-ffb2c9be64ba) (a hub of pre-trained models specifically for NLP tasks) is used to load the BERT pre-trained tokenizer and BERT pre-trained model.

# How to use the code? 
The code consists of the following modules:
1. **data_loader_and_preprocessor.py:** This script loads data from the dataset and performs some basic preprocessing on that data such as splitting it into train and test.
2.** dataset.py:** This is classes is inherited from Pytorh that can be easily iterated by the rest of the PyTorch and Pytorch lightning modules.
3. **data_module.py:** This is an inherited class from PyTorch lightning (a lightweight wrapper for PyTorch) that manages the training loop in a well-organized and effective way.
4. **train_valid_tagger.py:** This is also inherited from PyTorch lightning. The basic purpose of this class is to manage the overall logic of the model including model architecture, loss calculation, optimization, etc, and workflow of training, validation, and testing.
5. **train.py:** This file includes code that integrates all of the above modules and makes the model trainable.

# Prerequisite
The following libraries are required to be installed on your system before going to train the model. 
1. **Transformers:** pip install transformers
   For more details please visit ..... https://huggingface.co/models?library=transformers
3. **Pytroch:** pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (GPU), pip3 install torch torchvision torchaudio (CPU)
   For more details please visit ..... https://pytorch.org/get-started/locally/
5. **Pytorch Lightning:** pip install lightning
   For more details please visit ...... https://lightning.ai/docs/pytorch/stable/starter/installation.html
