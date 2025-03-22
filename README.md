
# Prothom Alo News Crawler, Summarizer, and Text-to-Speech Converter

## Overview

This project is a Python-based web scraper that fetches the latest news articles from the Prothom Alo website, summarizes them using a BERT-based model, and converts the summaries into audio using Google Text-to-Speech. The application is designed to handle Bengali text and provides a structured output in CSV format, including the original article content, summaries, and audio files.

## Features

- **Web Scraping**: Automatically fetches the latest articles from Prothom Alo.
- **Text Summarization**: Utilizes a BERT-based model to generate concise summaries of the articles.
- **Text-to-Speech**: Converts the summaries into audio files in Bengali using Google Text-to-Speech.
- **Data Storage**: Saves article URLs to avoid duplicate scraping and stores results in a CSV file.
- **Scheduled Scraping**: Runs at regular intervals (every 10 minutes) to keep the data updated.

## Requirements

- Python 3.6 or higher
- Libraries:
  - `requests`
  - `beautifulsoup4`
  - `csv`
  - `torch`
  - `transformers`
  - `numpy`
  - `scikit-learn`
  - `gtts`
  - `pandas`
  
You can install the required libraries using pip:

```bash
pip install requests beautifulsoup4 torch transformers numpy scikit-learn gtts pandas
```

## Setup

1. **Clone the Repository**: 
   Clone this repository to your local machine.

   ```bash
   git clone https://github.com/mdzubayerhossain/prothomalo_news_crawler_summarizer_tts.git
   cd prothomalo_news_crawler_summarizer_tts
   ```

2. **Install Dependencies**: 
   Make sure to install all the required libraries as mentioned above.

3. **Run the Application**: 
   Execute the main script to start the crawler.

   ```bash
   python main.py
   ```

## Usage

- The application will start scraping articles from Prothom Alo and will run indefinitely until stopped by the user (Ctrl+C).
- It will print logs to the console, indicating the progress of scraping, summarization, and audio generation.
- The results will be saved in a CSV file located in the `data` directory, along with audio files in the `data/audio` directory.

## Output

The output CSV file will contain the following columns:

- `title`: The title of the article.
- `full_content`: The full content of the article.
- `rag_summary`: The generated summary of the article.
- `image_url`: The URL of the main image associated with the article.
- `article_url`: The URL of the original article.
- `published_at`: The publication date of the article.
- `scraped_at`: The timestamp when the article was scraped.
- `audio_file`: The path to the generated audio file.

## Notes

- Ensure you have a stable internet connection while running the scraper.
- The scraping frequency can be adjusted by modifying the sleep duration in the main loop.
- Be mindful of the website's terms of service regarding web scraping.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- Thanks to the creators of the libraries used in this project, including `requests`, `BeautifulSoup`, `transformers`, and `gtts`.
- Special thanks to Prothom Alo for providing the news articles.

## Contact

For any questions or feedback, please reach out to [mdzubayerhossainpatowari@gmail.com].
```

Feel free to replace `https://github.com/mdzubayerhossain/prothomalo_news_crawler_summarizer_tts.git` and `prothomalo_news_crawler_summarizer_tts` with the actual URL of your repository and the directory name, respectively. Also, update the contact email with your actual email address.
