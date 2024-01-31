import streamlit as st
from bs4 import BeautifulSoup
import requests
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import pipeline

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
        st.error(f"Error occurred during scraping: {e}")
        return [], []

# Sample function for named entity recognition in Bangla
def perform_bangla_ner(text):
    try:
        tokenizer = BertTokenizerFast.from_pretrained("/content/drive/MyDrive/Colab_Notebooks/custom_model_transformers")
        model = BertForTokenClassification.from_pretrained("/content/drive/MyDrive/Colab_Notebooks/custom_model_transformers")
        nlp = pipeline("ner", model=model, tokenizer=tokenizer)

        # Process the text with the NER pipeline
        named_entities = nlp(text)

        # Extract entity text only
        entity_texts = [entity['word'] for entity in named_entities]

        # Return named entity texts
        return entity_texts

    except Exception as e:
        st.error(f"Error occurred during NER: {e}")
        return []

def main():
    st.title("Bangla News Entity Recognition")

    # Date input
    date = st.text_input("Enter Date (YYYY-MM-DD):")

    if st.button("Submit"):
        if date:
            headlines, articles = scrape_bangla_news(date)

            if headlines:
                st.subheader("Headlines:")
                for headline in headlines:
                    st.write(headline)

            if articles:
                st.subheader("Named Entities:")
                for article in articles:
                    named_entities = perform_bangla_ner(article)
                    st.write(named_entities)

if __name__ == "__main__":
    main()
