"""
Customer Sentiment Analysis Package

This package provides modules for scraping, preprocessing, analyzing, and visualizing
customer sentiment from product reviews.

Modules:
    - scraper: Web scraping functionality for Flipkart reviews
    - preprocessor: Data cleaning and text preprocessing
    - sentiment_analyzer: Sentiment analysis using TextBlob
    - visualizer: Data visualization and reporting
"""

__version__ = "1.0.0"
__author__ = "Gaganjit Singh"

from .scraper import FlipkartReviewScraper
from .preprocessor import ReviewPreprocessor
from .sentiment_analyzer import SentimentAnalyzer
from .visualizer import SentimentVisualizer

__all__ = [
    'FlipkartReviewScraper',
    'ReviewPreprocessor',
    'SentimentAnalyzer',
    'SentimentVisualizer'
]
