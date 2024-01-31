import spacy
import shutil
from transformers import BertTokenizer, BertConfig, BertForTokenClassification

# Save the trained model to a directory
trained_model_path = "Bangla_NEWS_NER/models/custom_model.spacy"
trained_model.to_disk(trained_model_path)

# Load the saved SpaCy model
nlp_transformers = spacy.load(trained_model_path)

# Save the model configuration
config = BertConfig.from_pretrained("bert-base-multilingual-cased")
config.save_pretrained("Bangla_NEWS_NER/models/custom_model_transformers")

# Initialize a BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Initialize a BERT model for token classification
model = BertForTokenClassification(config)

# Copy the weights from the SpaCy model to the BERT model
model.save_pretrained("Bangla_NEWS_NER/models/custom_model_transformers")

# Save the tokenizer in transformers format
tokenizer.save_pretrained("Bangla_NEWS_NER/models/custom_model_transformers")
