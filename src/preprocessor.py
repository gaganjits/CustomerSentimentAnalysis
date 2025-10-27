"""
Data Preprocessing Module
This module handles cleaning and preprocessing of scraped review data.
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class ReviewPreprocessor:
    """
    A class to preprocess and clean customer review data.
    """

    def __init__(self):
        """
        Initialize the preprocessor with necessary NLP tools.
        """
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def remove_duplicates(self, df):
        """
        Remove duplicate reviews from the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing review data

        Returns:
        --------
        pd.DataFrame : DataFrame with duplicates removed
        """
        initial_count = len(df)
        df = df.drop_duplicates(subset=['review_text'], keep='first')
        duplicates_removed = initial_count - len(df)

        print(f"Removed {duplicates_removed} duplicate reviews.")
        print(f"Remaining reviews: {len(df)}")

        return df.reset_index(drop=True)

    def handle_missing_values(self, df):
        """
        Handle missing values in the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing review data

        Returns:
        --------
        pd.DataFrame : DataFrame with missing values handled
        """
        print("\nMissing values before handling:")
        print(df.isnull().sum())

        # Remove rows where review_text is missing (essential field)
        df = df.dropna(subset=['review_text'])

        # Fill missing usernames with 'Anonymous'
        df['username'] = df['username'].fillna('Anonymous')

        # Fill missing ratings with median rating
        if df['rating'].isnull().any():
            median_rating = df['rating'].median()
            df['rating'] = df['rating'].fillna(median_rating)

        print("\nMissing values after handling:")
        print(df.isnull().sum())

        return df.reset_index(drop=True)

    def clean_text(self, text):
        """
        Clean individual review text by removing special characters and extra spaces.

        Parameters:
        -----------
        text : str
            Raw review text

        Returns:
        --------
        str : Cleaned review text
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_text(self, text):
        """
        Tokenize text into individual words.

        Parameters:
        -----------
        text : str
            Cleaned review text

        Returns:
        --------
        list : List of tokens
        """
        if not text:
            return []

        tokens = word_tokenize(text)
        return tokens

    def remove_stopwords(self, tokens):
        """
        Remove common stopwords from token list.

        Parameters:
        -----------
        tokens : list
            List of word tokens

        Returns:
        --------
        list : List of tokens with stopwords removed
        """
        filtered_tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        return filtered_tokens

    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens to their base form.

        Parameters:
        -----------
        tokens : list
            List of word tokens

        Returns:
        --------
        list : List of lemmatized tokens
        """
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens]
        return lemmatized

    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline for a single text.

        Parameters:
        -----------
        text : str
            Raw review text

        Returns:
        --------
        str : Preprocessed review text
        """
        # Clean text
        cleaned = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize_text(cleaned)

        # Remove stopwords
        filtered = self.remove_stopwords(tokens)

        # Lemmatize
        lemmatized = self.lemmatize_tokens(filtered)

        # Join back into string
        preprocessed_text = ' '.join(lemmatized)

        return preprocessed_text

    def preprocess_dataframe(self, df):
        """
        Apply complete preprocessing pipeline to entire DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing review data

        Returns:
        --------
        pd.DataFrame : Preprocessed DataFrame with new columns
        """
        print("Starting data preprocessing...")

        # Remove duplicates
        df = self.remove_duplicates(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Create cleaned text column (lowercase, no special chars)
        print("\nCleaning review text...")
        df['cleaned_text'] = df['review_text'].apply(self.clean_text)

        # Create preprocessed text column (cleaned + tokenized + stopwords removed + lemmatized)
        print("Preprocessing review text (tokenization, stopword removal, lemmatization)...")
        df['preprocessed_text'] = df['review_text'].apply(self.preprocess_text)

        # Add review length metrics
        df['review_length'] = df['review_text'].apply(len)
        df['word_count'] = df['review_text'].apply(lambda x: len(str(x).split()))

        print(f"\nPreprocessing completed!")
        print(f"Final dataset shape: {df.shape}")

        return df

    def get_preprocessing_summary(self, df):
        """
        Generate a summary of the preprocessing results.

        Parameters:
        -----------
        df : pd.DataFrame
            Preprocessed DataFrame

        Returns:
        --------
        dict : Summary statistics
        """
        summary = {
            'total_reviews': len(df),
            'unique_users': df['username'].nunique(),
            'avg_rating': df['rating'].mean(),
            'avg_review_length': df['review_length'].mean(),
            'avg_word_count': df['word_count'].mean(),
            'rating_distribution': df['rating'].value_counts().to_dict()
        }

        return summary


def save_preprocessed_data(df, filepath='data/preprocessed_reviews.csv'):
    """
    Save preprocessed reviews to a CSV file.

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
    filepath : str
        Path to save the CSV file
    """
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"\nPreprocessed data saved to {filepath}")


def load_preprocessed_data(filepath='data/preprocessed_reviews.csv'):
    """
    Load preprocessed reviews from a CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame : Preprocessed DataFrame
    """
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"Loaded {len(df)} preprocessed reviews from {filepath}")
    return df


# Example usage
if __name__ == "__main__":
    # Load raw reviews
    raw_df = pd.read_csv('data/raw_reviews.csv')

    # Initialize preprocessor
    preprocessor = ReviewPreprocessor()

    # Preprocess data
    processed_df = preprocessor.preprocess_dataframe(raw_df)

    # Get summary
    summary = preprocessor.get_preprocessing_summary(processed_df)
    print("\nPreprocessing Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Save preprocessed data
    save_preprocessed_data(processed_df)

    # Display sample
    print("\nSample of preprocessed data:")
    print(processed_df[['username', 'rating', 'review_text', 'preprocessed_text']].head())
