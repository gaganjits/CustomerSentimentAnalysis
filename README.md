# Customer Sentiment Analysis - iPhone 15 128GB

A comprehensive data analytics project that performs sentiment analysis on customer reviews from Flipkart to evaluate public perception of the iPhone 15 128GB model.

## Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Data Pipeline](#data-pipeline)
- [Results and Insights](#results-and-insights)
- [Sample Output](#sample-output)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project was developed for Amazon's Data Analytics team to understand customer sentiment towards the iPhone 15 128GB model by analyzing reviews from Flipkart. The project employs web scraping, natural language processing, and machine learning techniques to extract, process, and analyze customer feedback.

The analysis provides actionable insights about customer satisfaction, product strengths and weaknesses, and recommendations for improving marketing strategies and customer experience.

## Project Objectives

1. **Data Collection**: Scrape 300+ customer reviews from Flipkart using automated web scraping
2. **Data Preprocessing**: Clean and prepare text data for analysis using NLP techniques
3. **Sentiment Analysis**: Classify reviews as positive or negative using TextBlob
4. **Data Visualization**: Create comprehensive visualizations to understand sentiment patterns
5. **Insights Generation**: Extract actionable insights for business decision-making

## Features

- Automated web scraping of customer reviews using Selenium and BeautifulSoup
- Comprehensive data cleaning and preprocessing pipeline
- Advanced sentiment analysis with polarity and subjectivity scoring
- Beautiful and informative data visualizations
- Word cloud generation for positive and negative reviews
- Correlation analysis between ratings and sentiment
- Detailed reporting with actionable recommendations
- Modular and reusable code structure
- Jupyter Notebook with complete analysis workflow

## Technologies Used

### Programming Language
- Python 3.9+

### Libraries and Frameworks

**Web Scraping**
- Selenium 4.15.2 - Browser automation
- BeautifulSoup4 4.12.2 - HTML parsing
- webdriver-manager 4.0.1 - ChromeDriver management

**Data Processing**
- Pandas 2.1.3 - Data manipulation and analysis
- NumPy - Numerical operations

**Natural Language Processing**
- TextBlob 0.17.1 - Sentiment analysis
- NLTK 3.8.1 - Text preprocessing, tokenization, lemmatization

**Data Visualization**
- Matplotlib 3.8.2 - Plotting and visualization
- Seaborn 0.13.0 - Statistical visualizations
- WordCloud 1.9.3 - Word cloud generation

**Development**
- Jupyter Notebook 7.0.6 - Interactive development environment

## Project Structure

```
CustomerSentimentAnalysis/
│
├── data/                          # Data directory
│   ├── raw_reviews.csv           # Scraped raw reviews
│   ├── preprocessed_reviews.csv  # Cleaned and preprocessed data
│   ├── sentiment_analysis_results.csv  # Final analysis results
│   └── visualizations/           # Generated visualizations
│       ├── sentiment_distribution.png
│       ├── sentiment_by_rating.png
│       ├── polarity_distribution.png
│       ├── review_length_analysis.png
│       ├── wordcloud_positive.png
│       ├── wordcloud_negative.png
│       └── sentiment_categories.png
│
├── notebooks/                     # Jupyter notebooks
│   └── Customer_Sentiment_Analysis.ipynb
│
├── src/                          # Source code modules
│   ├── scraper.py               # Web scraping module
│   ├── preprocessor.py          # Data preprocessing module
│   ├── sentiment_analyzer.py    # Sentiment analysis module
│   └── visualizer.py            # Visualization module
│
├── requirements.txt              # Project dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # Project documentation
```

## Installation

### Prerequisites

- Python 3.9 or higher
- Google Chrome browser (for Selenium)
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/gaganjits/CustomerSentimentAnalysis.git
cd CustomerSentimentAnalysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Usage

### Option 1: Using Jupyter Notebook (Recommended)

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to `notebooks/Customer_Sentiment_Analysis.ipynb`

3. Run all cells sequentially to:
   - Scrape reviews from Flipkart
   - Preprocess the data
   - Perform sentiment analysis
   - Generate visualizations
   - View insights and recommendations

### Option 2: Using Individual Modules

#### 1. Web Scraping

```python
from src.scraper import FlipkartReviewScraper, save_reviews

# Initialize scraper
PRODUCT_URL = "https://www.flipkart.com/apple-iphone-15-black-128-gb/p/itm6d16e1cf03604"
scraper = FlipkartReviewScraper(PRODUCT_URL, headless=True)

# Scrape reviews
reviews_df = scraper.scrape_reviews(target_count=300, max_pages=30)

# Save to CSV
save_reviews(reviews_df, 'data/raw_reviews.csv')
```

#### 2. Data Preprocessing

```python
from src.preprocessor import ReviewPreprocessor, save_preprocessed_data
import pandas as pd

# Load raw reviews
df = pd.read_csv('data/raw_reviews.csv')

# Initialize preprocessor
preprocessor = ReviewPreprocessor()

# Preprocess data
processed_df = preprocessor.preprocess_dataframe(df)

# Save preprocessed data
save_preprocessed_data(processed_df, 'data/preprocessed_reviews.csv')
```

#### 3. Sentiment Analysis

```python
from src.sentiment_analyzer import SentimentAnalyzer, save_sentiment_results

# Load preprocessed data
df = pd.read_csv('data/preprocessed_reviews.csv')

# Initialize analyzer
analyzer = SentimentAnalyzer(polarity_threshold=0.1)

# Perform sentiment analysis
sentiment_df = analyzer.analyze_dataframe(df)

# Get summary
summary = analyzer.get_sentiment_summary(sentiment_df)

# Save results
save_sentiment_results(sentiment_df, 'data/sentiment_analysis_results.csv')
```

#### 4. Visualization

```python
from src.visualizer import SentimentVisualizer

# Load sentiment results
df = pd.read_csv('data/sentiment_analysis_results.csv')

# Initialize visualizer
visualizer = SentimentVisualizer()

# Create all visualizations
visualizer.create_comprehensive_dashboard(df, output_dir='data/visualizations')
```

## Modules

### 1. scraper.py - Web Scraping Module

**Key Features:**
- Automated browser control with Selenium
- Dynamic page navigation and pagination handling
- Robust error handling and retry mechanisms
- Extract username, rating, and review text
- Configurable scraping parameters

**Main Class:** `FlipkartReviewScraper`

**Key Methods:**
- `scrape_reviews()` - Main scraping function
- `extract_reviews_from_page()` - Extract reviews from current page
- `click_next_page()` - Navigate to next page

### 2. preprocessor.py - Data Preprocessing Module

**Key Features:**
- Remove duplicate reviews
- Handle missing values
- Text cleaning (lowercase, remove special characters)
- Tokenization
- Stop word removal
- Lemmatization
- Review length metrics

**Main Class:** `ReviewPreprocessor`

**Key Methods:**
- `preprocess_dataframe()` - Complete preprocessing pipeline
- `clean_text()` - Clean individual text
- `remove_stopwords()` - Filter stop words
- `lemmatize_tokens()` - Lemmatize tokens

### 3. sentiment_analyzer.py - Sentiment Analysis Module

**Key Features:**
- TextBlob-based sentiment analysis
- Polarity and subjectivity scoring
- Binary sentiment classification (Positive/Negative)
- Detailed sentiment categories (Very Positive, Positive, Neutral, Negative, Very Negative)
- Correlation analysis between ratings and sentiment
- Extreme review identification

**Main Class:** `SentimentAnalyzer`

**Key Methods:**
- `analyze_dataframe()` - Analyze entire dataset
- `get_sentiment_scores()` - Calculate polarity and subjectivity
- `classify_sentiment()` - Classify as positive or negative
- `analyze_sentiment_by_rating()` - Group analysis by rating

### 4. visualizer.py - Data Visualization Module

**Key Features:**
- Sentiment distribution charts
- Sentiment by rating analysis
- Polarity distribution plots
- Review length analysis
- Word clouds for positive and negative reviews
- Detailed sentiment category breakdown

**Main Class:** `SentimentVisualizer`

**Key Methods:**
- `plot_sentiment_distribution()` - Bar and pie charts
- `plot_sentiment_by_rating()` - Rating correlation plots
- `create_wordcloud()` - Generate word clouds
- `create_comprehensive_dashboard()` - Generate all visualizations

## Data Pipeline

The project follows a systematic data pipeline:

```
1. Data Collection
   └─> Web Scraping (Selenium + BeautifulSoup)
       └─> raw_reviews.csv

2. Data Preprocessing
   └─> Text Cleaning (Pandas + NLTK)
       └─> preprocessed_reviews.csv

3. Sentiment Analysis
   └─> Sentiment Scoring (TextBlob)
       └─> sentiment_analysis_results.csv

4. Visualization
   └─> Generate Charts and Word Clouds
       └─> data/visualizations/

5. Insights & Recommendations
   └─> Business Intelligence Report
```

## Results and Insights

The sentiment analysis provides the following key insights:

### Sentiment Distribution
- Overall percentage of positive vs negative reviews
- Average sentiment polarity score
- Average subjectivity score

### Rating Analysis
- Correlation between numeric ratings and sentiment polarity
- Sentiment distribution across different star ratings
- Identification of rating-sentiment mismatches

### Common Themes
- Most frequently mentioned words in positive reviews
- Most frequently mentioned words in negative reviews
- Key product features highlighted by customers

### Review Characteristics
- Average review length by sentiment
- Relationship between review detail and sentiment intensity

### Actionable Recommendations
- Marketing strategy improvements
- Product page optimization suggestions
- Customer experience enhancements
- Inventory and pricing insights

## Sample Output

### Sentiment Summary Example

```
=============================================================
SENTIMENT ANALYSIS REPORT
=============================================================

Total Reviews Analyzed: 300

Sentiment Distribution:
  Positive: 245 (81.67%)
  Negative: 55 (18.33%)

Average Sentiment Scores:
  Polarity: 0.287 (std: 0.312)
  Subjectivity: 0.542

Detailed Sentiment Categories:
  Very Positive: 98
  Positive: 147
  Neutral: 23
  Negative: 28
  Very Negative: 4
=============================================================
```

### Key Findings
- Strong positive sentiment indicates high customer satisfaction
- Positive reviews highlight camera quality, performance, and design
- Negative reviews mention battery life, pricing, and heating issues
- High correlation between ratings and sentiment validates review authenticity

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Amazon Data Analytics Team for project requirements
- Flipkart for providing publicly available review data
- TextBlob developers for sentiment analysis capabilities
- Open-source community for excellent Python libraries

## Contact

For questions or feedback, please open an issue on GitHub.

## Project Status

This project is complete and ready for submission. All requirements have been implemented:

- [x] Data Collection (300+ reviews scraped)
- [x] Data Cleaning and Preprocessing
- [x] Sentiment Analysis with TextBlob
- [x] Comprehensive Data Visualizations
- [x] Detailed Analysis Report
- [x] Well-commented code
- [x] Professional documentation
- [x] Jupyter Notebook with sample scenarios

---

**Note**: This project is for educational and analytical purposes. Always respect website terms of service and robots.txt when performing web scraping.
