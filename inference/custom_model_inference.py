# Required libraries
from bs4 import BeautifulSoup
import requests
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from transformers import pipeline
import datetime

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Sample URL for Bangla news scraping
NEWS_URL = "https://www.prothomalony.com/"

# Sample function to scrape Bangla news headlines and articles
def scrape_bangla_news(date):
    try:
        url = f"{NEWS_URL}?date={date}"

        # Send GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract headlines and articles (you need to adjust this based on the website's HTML structure)
        headlines = [headline.text for headline in soup.find_all("h2")]
        articles = [article.text for article in soup.find_all("div")]

        # Return scraped headlines and articles
        return headlines, articles

    except Exception as e:
        print("Error occurred during scraping:", e)
        return [], []

# Sample function for named entity recognition in Bangla
def perform_bangla_ner(text):
    try:
        tokenizer = BertTokenizerFast.from_pretrained("Bangla_NEWS_NER/models/custom_model_transformers")
        model = BertForTokenClassification.from_pretrained("Bangla_NEWS_NER/models/custom_model_transformers")
        nlp = pipeline("ner", model=model, tokenizer=tokenizer)

        # Process the text with the NER pipeline
        named_entities = nlp(text)

        # Extract entity text only
        entity_texts = [entity['word'] for entity in named_entities]

        # Return named entity texts
        return entity_texts

    except Exception as e:
        print("Error occurred during NER:", e)
        return []

# Sample function to toggle date selection from the backend
def select_date():
    # Assuming you have a backend function to retrieve the desired date
    # This function returns the selected date
    selected_date = "2024-01-27"  # Example date

    return selected_date

# Main function to orchestrate the process
def main():
    try:
        # Toggle date selection
        selected_date = select_date()

        # Scrape Bangla news headlines and articles
        headlines, articles = scrape_bangla_news(selected_date)

        # Perform named entity recognition on the articles
        for article in articles:
            named_entities = perform_bangla_ner(article)
            print("Named Entities:", named_entities)

    except Exception as e:
        print("Error occurred:", e)

# Entry point
if __name__ == "__main__":
    main()