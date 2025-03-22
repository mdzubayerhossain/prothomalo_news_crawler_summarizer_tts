import requests
from bs4 import BeautifulSoup
import csv
import time
import random
import re
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from gtts import gTTS  # Import Google Text-to-Speech

def clean_text(text):
    """Clean text by removing extra whitespace and newlines"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_current_articles():
    """Get list of article URLs already scraped"""
    existing_urls = set()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Check if history file exists
    if os.path.exists('data/article_history.txt'):
        with open('data/article_history.txt', 'r', encoding='utf-8') as f:
            for line in f:
                existing_urls.add(line.strip())
    
    return existing_urls

def save_article_url(url):
    """Save article URL to history file"""
    with open('data/article_history.txt', 'a', encoding='utf-8') as f:
        f.write(url + '\n')

def get_sentence_embeddings(sentences, model, tokenizer):
    """Get embeddings for a list of sentences using BERT"""
    embeddings = []
    
    for sentence in sentences:
        # Tokenize and convert to tensor
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use CLS token as sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(embedding[0])
    
    return np.array(embeddings)

def split_into_sentences(text):
    """Split Bengali text into sentences"""
    # Simple rule-based sentence splitting for Bengali
    # This is a basic implementation and might need improvement
    sentences = re.split(r'[।!?]', text)
    return [s.strip() for s in sentences if s.strip()]

def summarize_bengali_with_rag(text, model, tokenizer, num_sentences=3):
    """
    Summarize Bengali text using a RAG-inspired approach:
    1. Split text into sentences
    2. Get sentence embeddings
    3. Calculate sentence importance based on centrality
    4. Select top sentences maintaining original order
    """
    if not text or len(text.strip()) == 0:
        return ""
    
    # Clean text
    text = clean_text(text)
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    # If text is already short, return as is
    if len(sentences) <= num_sentences:
        return text
    
    # Get sentence embeddings
    embeddings = get_sentence_embeddings(sentences, model, tokenizer)
    
    # Calculate sentence centrality (similarity to other sentences)
    similarity_matrix = cosine_similarity(embeddings)
    centrality_scores = np.sum(similarity_matrix, axis=1)
    
    # Get indices of top sentences by centrality
    top_indices = np.argsort(centrality_scores)[-num_sentences:]
    
    # Sort indices to maintain original order
    top_indices = sorted(top_indices)
    
    # Construct summary from selected sentences
    selected_sentences = [sentences[i] for i in top_indices]
    summary = '। '.join(selected_sentences) + '।'
    
    return summary

def text_to_speech_bengali(text, filename):
    """
    Convert Bengali text to speech using Google Text-to-Speech
    Returns the path to the saved audio file
    """
    # Create audio directory if it doesn't exist
    audio_dir = 'data/audio'
    os.makedirs(audio_dir, exist_ok=True)
    
    # Generate audio file path
    file_path = os.path.join(audio_dir, filename)
    
    try:
        # Create gTTS object with Bengali language
        tts = gTTS(text=text, lang='bn', slow=False)
        
        # Save to file
        tts.save(file_path)
        
        print(f"Audio saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error creating audio: {e}")
        return ""

def scrape_and_summarize_prothom_alo(model, tokenizer):
    """Scrape latest news articles from Prothom Alo and summarize them"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting scrape cycle...")
    
    # Get already scraped article URLs
    existing_urls = get_current_articles()
    
    # Headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,bn;q=0.8",
    }
    
    # Base URL and homepage URL
    base_url = "https://www.prothomalo.com"
    home_url = base_url
    
    # Create or append to CSV file
    csv_filename = f'data/prothom_alo_with_summaries_{datetime.now().strftime("%Y%m%d")}.csv'
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write header only if file is new
        if not file_exists:
            writer.writerow([
                'title', 'full_content', 'rag_summary', 'image_url', 
                'article_url', 'published_at', 'scraped_at', 'audio_file'
            ])
        
        try:
            print("Fetching homepage to extract article links...")
            response = requests.get(home_url, headers=headers)
            response.raise_for_status()
            
            # Parse homepage HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find article links on the homepage
            article_links = []
            categories = ['bangladesh', 'world', 'economy', 'sports', 'entertainment', 'opinion', 'lifestyle', 'technology']
            
            # Look for article cards or links in major categories
            for category in categories:
                category_selector = f'a[href*="/{category}/"]'
                category_links = soup.select(category_selector)
                
                for element in category_links:
                    href = element.get('href')
                    if href and '/video/' not in href and '/gallery/' not in href:
                        # Make sure it's a full URL
                        if not href.startswith('http'):
                            href = base_url + href
                        article_links.append(href)
            
            # Filter out already scraped articles
            new_articles = [url for url in set(article_links) if url not in existing_urls]
            
            if not new_articles:
                print("No new articles found in this cycle.")
                return 0
                
            print(f"Found {len(new_articles)} new article links. Processing...")
            
            # Process each article
            articles_scraped = 0
            for i, article_url in enumerate(new_articles):
                try:
                    # Add a random delay to avoid being blocked
                    time.sleep(random.uniform(2, 3))
                    
                    print(f"Fetching article {i+1}/{len(new_articles)}: {article_url}")
                    article_response = requests.get(article_url, headers=headers)
                    article_response.raise_for_status()
                    
                    # Parse article HTML
                    article_soup = BeautifulSoup(article_response.text, 'html.parser')
                    
                    # Extract title
                    title_element = article_soup.select_one('h1')
                    title = clean_text(title_element.text) if title_element else "No title found"
                    
                    # Extract publication date
                    date_element = article_soup.select_one('time')
                    published_at = date_element.get('datetime') if date_element else ""
                    
                    # Extract main image
                    image_element = article_soup.select_one('figure img')
                    image_url = ""
                    if image_element:
                        image_url = image_element.get('src')
                        if not image_url:
                            image_url = image_element.get('data-src', '')
                    
                    # Extract content from story-element-text elements
                    article_content = ""
                    
                    # Find all elements with class 'story-element-text'
                    story_elements = article_soup.select('.story-element-text')
                    
                    if story_elements:
                        for element in story_elements:
                            # Extract text from each story element
                            element_text = element.get_text(strip=True)
                            if element_text:
                                article_content += element_text + "\n\n"
                    
                    # Clean up the content
                    article_content = article_content.strip()
                    
                    if not article_content:
                        print(f"No story-element-text found for article: {article_url}")
                        # Fallback: Try another common content selector
                        fallback_elements = article_soup.select('.storyContent p')
                        if fallback_elements:
                            for p in fallback_elements:
                                article_content += p.get_text(strip=True) + "\n\n"
                            article_content = article_content.strip()
                            print(f"Used fallback method and found {len(article_content)} characters")
                        else:
                            article_content = "Content extraction failed - no story-element-text found"
                    
                    # Current timestamp
                    scraped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Generate RAG summary for the article
                    print(f"Generating RAG summary for article {i+1}...")
                    rag_summary = ""
                    if article_content and article_content != "Content extraction failed - no story-element-text found":
                        rag_summary = summarize_bengali_with_rag(article_content, model, tokenizer)
                        print(f"Summary generated ({len(rag_summary)} characters)")
                    else:
                        rag_summary = "Unable to generate summary - no content available"
                    
                    # Generate audio file for the title and summary
                    audio_file_path = ""
                    if rag_summary and title and rag_summary != "Unable to generate summary - no content available":
                        # Create audio content with title and summary
                        audio_text = f"{title}. {rag_summary}"
                        
                        # Create a filename based on article details
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        # Remove any characters that might be problematic in filenames
                        safe_title = re.sub(r'[^\w\s]', '', title[:30])
                        safe_title = re.sub(r'\s+', '_', safe_title)
                        audio_filename = f"{timestamp}_{safe_title}.mp3"
                        
                        print(f"Generating audio for article {i+1}...")
                        audio_file_path = text_to_speech_bengali(audio_text, audio_filename)
                    
                    # Write to CSV
                    writer.writerow([
                        title, article_content, rag_summary, image_url, 
                        article_url, published_at, scraped_at, audio_file_path
                    ])
                    
                    # Add to history
                    save_article_url(article_url)
                    
                    print(f"Saved article with summary and audio successfully")
                    articles_scraped += 1
                    
                except Exception as e:
                    print(f"Error processing article {article_url}: {e}")
                    
            return articles_scraped
                    
        except Exception as e:
            print(f"Error during scraping: {e}")
            return 0

def main():
    print("Starting Prothom Alo News Crawler, Summarizer, and Text-to-Speech Converter")
    print("Press Ctrl+C to stop the program")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Load model and tokenizer for summarization - using multilingual BERT
    print("Loading BERT model for summarization...")
    model_name = "bert-base-multilingual-cased"  # Supports Bengali
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    print("Model loaded successfully")
    
    # Keep track of statistics
    total_articles = 0
    cycles = 0
    
    try:
        while True:
            # Run the scraper and summarizer
            new_articles = scrape_and_summarize_prothom_alo(model, tokenizer)
            total_articles += new_articles
            cycles += 1
            
            # Print statistics
            print(f"\nCycle {cycles} completed.")
            print(f"Total articles scraped, summarized, and converted to audio so far: {total_articles}")
            print(f"Next scrape scheduled at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Wait for 10 minutes
            print("Waiting for 10 minutes before next scrape cycle...")
            for i in range(10):
                time.sleep(60)  # Wait 1 minute
                minutes_left = 9 - i
                if minutes_left > 0:
                    print(f"{minutes_left} minutes remaining until next scrape...")
    
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
        print(f"Total articles scraped, summarized, and converted to audio: {total_articles}")
        print(f"Total scrape cycles completed: {cycles}")
        print("Exiting...")

if __name__ == "__main__":
    main()
