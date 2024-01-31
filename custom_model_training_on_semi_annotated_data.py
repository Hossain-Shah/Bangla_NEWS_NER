import spacy
import requests
from bs4 import BeautifulSoup
import random
import datetime
from spacy.training.example import Example
from spacy.tokens import Span


# Step 1: Scraping Articles from Prothom Alo
def scrape_prothom_alo_articles():
    articles = []
    base_url = "https://www.prothomalony.com/"
    today = datetime.date.today()
    formatted_date = today.strftime("%Y-%m-%d")
    url = f"{base_url}?date={formatted_date}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # Extract headlines and articles (you need to adjust this based on the website's HTML structure)
        articles = [article.text for article in soup.find_all("div")]
        return articles

# Step 2: Preparing the Dataset
def prepare_dataset(articles, entities_file):
    nlp = spacy.blank("bn")
    ner = nlp.add_pipe("ner")
    ner.add_label("B-PER")
    ner.add_label("I-PER")
    dataset = []

    # Read entities from the file
    with open(entities_file, 'r', encoding='utf-8') as f:
        entities_data = f.readlines()

    for article, entities_str in zip(articles, entities_data):
        text = article
        entities = eval(entities_str)  # Assuming entities are stored as a string representation of a list of tuples

        # Check for overlapping entities and adjust if necessary
        non_overlapping_entities = []
        for start, end, label in entities:
            overlapping = False
            for existing_start, existing_end, _ in non_overlapping_entities:
                if start < existing_end and end > existing_start:
                    overlapping = True
                    break
            if not overlapping:
                non_overlapping_entities.append((start, end, label))

        dataset.append((text, {"entities": non_overlapping_entities}))

    return dataset


# Step 3: Training the SpaCy Model
def train_spacy_ner(dataset, nlp):
    # Convert (text, annotation) tuples to Example objects
    examples = []
    for text, annotations in dataset:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        examples.append(example)

    # Update the model with the Example objects
    losses = {}
    for epoch in range(10):
        random.shuffle(examples)
        for batch in spacy.util.minibatch(examples, size=8):
            nlp.update(batch, losses=losses, drop=0.5)
        print("Losses:", losses)

    return nlp

# Step 4: Showing Entities Date-wise
def show_entities_datewise(model, articles):
    date_entities = {}
    for article in articles:
        text = article
        doc = model(text)
        spans = []
        for ent in doc.spans:
            spans.append(Span(doc, ent.start, ent.end, ent.label))
        date_entities[formatted_date] = spans
    return date_entities

if __name__ == "__main__":
    # Create a SpaCy NLP object
    nlp = spacy.blank("bn")

    # Step 1: Scraping Articles from Prothom Alo
    articles = scrape_prothom_alo_articles()
    print(articles)

    # Provide path to the text file containing entities
    entities_file = "/content/drive/MyDrive/Colab_Notebooks/annotations.txt"

    # Step 2: Preparing the Dataset
    dataset = prepare_dataset(articles, entities_file)

    # Step 3: Training the SpaCy Model
    trained_model = train_spacy_ner(dataset, nlp)

    # Step 4: Showing Entities Date-wise
    formatted_date = datetime.date.today().strftime("%Y-%m-%d")
    date_entities = show_entities_datewise(trained_model, articles)
    if formatted_date in date_entities:
      print(f"Entities on {formatted_date}: {date_entities[formatted_date]}")
    else:
      print(f"No entities found for {formatted_date}")
    # Save the trained model
    trained_model.to_disk("/content/drive/MyDrive/Colab_Notebooks/custom_model.spacy")
