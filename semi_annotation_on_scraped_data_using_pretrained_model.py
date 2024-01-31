from bs4 import BeautifulSoup
import requests
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertTokenizerFast, BertConfig, BertForTokenClassification
from transformers import pipeline

# Load the tokenizer and model from Hugging Face Transformers
tokenizer = BertTokenizerFast.from_pretrained("saiful9379/BanglaNER_BERT")
model = BertForTokenClassification.from_pretrained("saiful9379/BanglaNER_BERT")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

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
        # Process the text with the NER pipeline
        named_entities = nlp(text)

        # Extract entity spans
        entity_spans = []
        for entity in named_entities:
            start_idx = text.find(entity["word"], entity["start"])
            end_idx = start_idx + len(entity["word"])
            entity_spans.append((start_idx, end_idx, entity["entity"]))

        # Return entity spans
        return entity_spans

    except Exception as e:
        print("Error occurred during NER:", e)
        return []

# Main function to orchestrate the process
def main():
    try:
        # Toggle date selection
        selected_date = select_date()

        # Scrape Bangla news headlines and articles
        headlines, articles = scrape_bangla_news(selected_date)

        # Save entity spans to a text file
        with open("/content/drive/MyDrive/Colab_Notebooks/annotations.txt", "w", encoding="utf-8") as file:
            for article in articles:
                entity_spans = perform_bangla_ner(article)
                print(entity_spans)
                for start, end, entity in entity_spans:
                    file.write(f"{start},{end},{entity}\n")

        print("Entity spans saved to annotations.txt")

    except Exception as e:
        print("Error occurred:", e)

# Sample function to toggle date selection from the backend
def select_date():
    # Assuming you have a backend function to retrieve the desired date
    # This function returns the selected date
    selected_date = "2024-01-29"  # Example date

    return selected_date

# Entry point
if __name__ == "__main__":
    main()
