"""
Sentiment Analysis Module
This module performs sentiment analysis on customer reviews using TextBlob.
"""

import pandas as pd
from textblob import TextBlob
import numpy as np


class SentimentAnalyzer:
    """
    A class to perform sentiment analysis on customer reviews.
    """

    def __init__(self, polarity_threshold=0.1):
        """
        Initialize the sentiment analyzer.

        Parameters:
        -----------
        polarity_threshold : float
            Threshold for classifying sentiment (default: 0.1)
            Polarity >= threshold: Positive
            Polarity < threshold: Negative
        """
        self.polarity_threshold = polarity_threshold

    def get_sentiment_scores(self, text):
        """
        Calculate sentiment polarity and subjectivity scores for a text.

        Parameters:
        -----------
        text : str
            Review text to analyze

        Returns:
        --------
        tuple : (polarity, subjectivity)
            polarity: float between -1 (negative) and 1 (positive)
            subjectivity: float between 0 (objective) and 1 (subjective)
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0, 0.0

        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            return polarity, subjectivity
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return 0.0, 0.0

    def classify_sentiment(self, polarity):
        """
        Classify sentiment based on polarity score.

        Parameters:
        -----------
        polarity : float
            Polarity score from TextBlob

        Returns:
        --------
        str : Sentiment classification ('Positive' or 'Negative')
        """
        if polarity >= self.polarity_threshold:
            return 'Positive'
        else:
            return 'Negative'

    def get_sentiment_category(self, polarity):
        """
        Get detailed sentiment category based on polarity score.

        Parameters:
        -----------
        polarity : float
            Polarity score from TextBlob

        Returns:
        --------
        str : Detailed sentiment category
        """
        if polarity >= 0.5:
            return 'Very Positive'
        elif polarity >= 0.1:
            return 'Positive'
        elif polarity > -0.1:
            return 'Neutral'
        elif polarity > -0.5:
            return 'Negative'
        else:
            return 'Very Negative'

    def analyze_dataframe(self, df, text_column='review_text'):
        """
        Perform sentiment analysis on entire DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing review data
        text_column : str
            Name of column containing review text (default: 'review_text')

        Returns:
        --------
        pd.DataFrame : DataFrame with sentiment analysis results added
        """
        print("Performing sentiment analysis on reviews...")

        # Calculate sentiment scores
        sentiment_scores = df[text_column].apply(self.get_sentiment_scores)

        df['polarity'] = sentiment_scores.apply(lambda x: x[0])
        df['subjectivity'] = sentiment_scores.apply(lambda x: x[1])

        # Classify sentiment
        df['sentiment'] = df['polarity'].apply(self.classify_sentiment)

        # Add detailed sentiment category
        df['sentiment_category'] = df['polarity'].apply(self.get_sentiment_category)

        print("Sentiment analysis completed!")

        return df

    def get_sentiment_summary(self, df):
        """
        Generate summary statistics for sentiment analysis results.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with sentiment analysis results

        Returns:
        --------
        dict : Summary statistics
        """
        summary = {
            'total_reviews': len(df),
            'positive_count': len(df[df['sentiment'] == 'Positive']),
            'negative_count': len(df[df['sentiment'] == 'Negative']),
            'positive_percentage': (len(df[df['sentiment'] == 'Positive']) / len(df)) * 100,
            'negative_percentage': (len(df[df['sentiment'] == 'Negative']) / len(df)) * 100,
            'avg_polarity': df['polarity'].mean(),
            'avg_subjectivity': df['subjectivity'].mean(),
            'polarity_std': df['polarity'].std(),
            'sentiment_by_category': df['sentiment_category'].value_counts().to_dict()
        }

        return summary

    def analyze_sentiment_by_rating(self, df):
        """
        Analyze sentiment distribution across different ratings.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with sentiment analysis results

        Returns:
        --------
        pd.DataFrame : Summary of sentiment by rating
        """
        sentiment_by_rating = df.groupby('rating').agg({
            'polarity': ['mean', 'std', 'count'],
            'subjectivity': 'mean',
            'sentiment': lambda x: (x == 'Positive').sum() / len(x) * 100
        }).round(3)

        sentiment_by_rating.columns = ['avg_polarity', 'polarity_std', 'count', 'avg_subjectivity', 'positive_percentage']

        return sentiment_by_rating

    def get_extreme_reviews(self, df, n=5):
        """
        Get the most positive and most negative reviews.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with sentiment analysis results
        n : int
            Number of reviews to return for each category (default: 5)

        Returns:
        --------
        dict : Dictionary with most positive and negative reviews
        """
        most_positive = df.nlargest(n, 'polarity')[['username', 'rating', 'review_text', 'polarity', 'sentiment']]
        most_negative = df.nsmallest(n, 'polarity')[['username', 'rating', 'review_text', 'polarity', 'sentiment']]

        return {
            'most_positive': most_positive,
            'most_negative': most_negative
        }

    def analyze_correlation(self, df):
        """
        Analyze correlation between rating and sentiment polarity.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with sentiment analysis results

        Returns:
        --------
        float : Correlation coefficient between rating and polarity
        """
        correlation = df['rating'].corr(df['polarity'])
        print(f"\nCorrelation between rating and sentiment polarity: {correlation:.3f}")

        return correlation


def save_sentiment_results(df, filepath='data/sentiment_analysis_results.csv'):
    """
    Save sentiment analysis results to a CSV file.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sentiment analysis results
    filepath : str
        Path to save the CSV file
    """
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"\nSentiment analysis results saved to {filepath}")


def load_sentiment_results(filepath='data/sentiment_analysis_results.csv'):
    """
    Load sentiment analysis results from a CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame : DataFrame with sentiment analysis results
    """
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"Loaded {len(df)} reviews with sentiment analysis from {filepath}")
    return df


def print_sentiment_report(summary):
    """
    Print a formatted sentiment analysis report.

    Parameters:
    -----------
    summary : dict
        Summary statistics from sentiment analysis
    """
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS REPORT")
    print("="*60)
    print(f"\nTotal Reviews Analyzed: {summary['total_reviews']}")
    print(f"\nSentiment Distribution:")
    print(f"  Positive: {summary['positive_count']} ({summary['positive_percentage']:.2f}%)")
    print(f"  Negative: {summary['negative_count']} ({summary['negative_percentage']:.2f}%)")
    print(f"\nAverage Sentiment Scores:")
    print(f"  Polarity: {summary['avg_polarity']:.3f} (std: {summary['polarity_std']:.3f})")
    print(f"  Subjectivity: {summary['avg_subjectivity']:.3f}")
    print(f"\nDetailed Sentiment Categories:")
    for category, count in sorted(summary['sentiment_by_category'].items()):
        print(f"  {category}: {count}")
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_reviews.csv')

    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer(polarity_threshold=0.1)

    # Perform sentiment analysis
    df_with_sentiment = analyzer.analyze_dataframe(df)

    # Get summary
    summary = analyzer.get_sentiment_summary(df_with_sentiment)
    print_sentiment_report(summary)

    # Analyze sentiment by rating
    print("\nSentiment Analysis by Rating:")
    sentiment_by_rating = analyzer.analyze_sentiment_by_rating(df_with_sentiment)
    print(sentiment_by_rating)

    # Analyze correlation
    correlation = analyzer.analyze_correlation(df_with_sentiment)

    # Get extreme reviews
    extreme_reviews = analyzer.get_extreme_reviews(df_with_sentiment, n=3)
    print("\nMost Positive Reviews:")
    print(extreme_reviews['most_positive'][['username', 'rating', 'polarity']])
    print("\nMost Negative Reviews:")
    print(extreme_reviews['most_negative'][['username', 'rating', 'polarity']])

    # Save results
    save_sentiment_results(df_with_sentiment)
