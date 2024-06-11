# Sentiment Analysis with Deep Learning using BERT
## Project Outline
1. Introduction
2. Exploratory Data Analysis and Preprocessing
3. Training/Validation Split
4. Loading Tokenizer and Encoding our Data
5. Setting up BERT Pre-trained Model
6. Creating Data Loaders
7. Setting Up Optimizer and Scheduler
8. Defining our Performance Metrics
9. Creating our Training Loop

## Introduction
### What is BERT?
BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art language model developed by Google. It uses transformer architecture to create bidirectional representations, meaning it considers the context from both directions (left and right) for understanding the meaning of a word. BERT can be fine-tuned for a variety of natural language processing tasks such as question answering, sentiment analysis, and more.

For more information, refer to the [original BERT paper](https://arxiv.org/abs/1810.04805) and the [HuggingFace documentation](https://huggingface.co/docs/transformers/model_doc/bert).

## Exploratory Data Analysis and Preprocessing
We will use the SMILE Twitter dataset for this project. This dataset contains annotated tweets categorized into different emotional states. The initial preprocessing steps involve loading the dataset, removing unwanted categories (like those with multiple labels), and encoding the remaining categories into numeric labels for model training.

## Training/Validation Split
To ensure our model generalizes well, we split the data into training and validation sets. This split allows us to train the model on one portion of the data and validate its performance on another, unseen portion. Stratified splitting is used to maintain the distribution of categories in both sets.

## Loading Tokenizer and Encoding our Data
We use the BertTokenizer from the HuggingFace library to convert our text data into the format required by BERT. This process includes:
-  Adding special tokens to mark the beginning and end of sentences.
- Padding or truncating sentences to a uniform length.
- Creating attention masks to indicate which tokens are actual data and which are padding.
The encoded data is then organized into tensors suitable for input into the BERT model.

## Setting up BERT Pretrained Model
We initialize a pre-trained BERT model for sequence classification using the BertForSequenceClassification class. This model is pre-trained on a large corpus of text and can be fine-tuned for specific tasks such as sentiment analysis by adjusting its parameters during training.

## Creating Data Loaders
Data loaders are created to handle batching and shuffling of data during training and validation. The DataLoader class from PyTorch is used to create iterators for our training and validation datasets. This helps in efficiently feeding data into the model in manageable batches.

## Setting Up Optimizer and Scheduler
The optimizer is responsible for updating the model's parameters based on the gradients computed during backpropagation. We use the AdamW optimizer, which is well-suited for transformer models. Additionally, a learning rate scheduler is set up to adjust the learning rate during training, helping the model converge more effectively.

## Defining our Performance Metrics
To evaluate the model's performance, we use metrics such as the F1 score and accuracy. The F1 score provides a balance between precision and recall, making it a good metric for imbalanced datasets. Accuracy per class is also calculated to understand how well the model performs on each specific category.

## Creating our Training Loop
The training loop involves:

1. Training Phase:
  - The model is set to training mode.
  - Batches of data are fed into the model, and the loss is calculated.
  - Gradients are computed, and the model's parameters are updated.
2. Evaluation Phase:
  - The model is set to evaluation mode.
  - Performance metrics are calculated on the validation set.
After each epoch, the model's state is saved. The final performance is assessed by loading the trained model and calculating accuracy per class.

This structured approach ensures that the model is trained effectively and its performance is evaluated thoroughly, enabling us to fine-tune and improve the model iteratively.
