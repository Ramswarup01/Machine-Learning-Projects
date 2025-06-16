# BERT-Based Sentiment Analysis (Fine-Tuning Project)

This project fine-tunes a pre-trained BERT model (`bert-base-uncased`) for binary sentiment classification (positive vs. negative) using a custom movie review dataset.

## 📁 Dataset

- The dataset used is the IMDb movie reviews dataset for sentiment analysis.
- You can download it here: [IMDb Dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
- After downloading, extract the archive and organize the data into `train.csv` and `test.csv` files with `text` and `sentiment` columns (where sentiment is `pos` or `neg`).

## 🚀 Features

- Uses Hugging Face Transformers for model and tokenizer
- Fine-tunes on a subset (20%) of the dataset for faster iteration
- Evaluates model using accuracy, precision, recall, and F1-score
- Saves the fine-tuned model for future use
- Includes a prediction function for new texts

## 🧰 Dependencies

Install required packages:

```bash
pip install torch transformers pandas numpy scikit-learn
```
## Fine-tuned Model

The fine-tuned model is saved and shared here:  
[Google Drive - Fine-tuned BERT Model](https://drive.google.com/drive/folders/1IahaGayIH1vea4MHvpxm-nXVUNuWUsZP?usp=sharing)

To load the model:

```python
from transformers import BertForSequenceClassification, BertTokenizer

model_path = 'path_to_downloaded_model_folder'  # Replace with your local path

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
