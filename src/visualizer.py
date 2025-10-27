"""
Data Visualization Module
This module creates visualizations for sentiment analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class SentimentVisualizer:
    """
    A class to create visualizations for sentiment analysis results.
    """

    def __init__(self, figsize=(12, 6)):
        """
        Initialize the visualizer.

        Parameters:
        -----------
        figsize : tuple
            Default figure size for plots (default: (12, 6))
        """
        self.figsize = figsize

    def plot_sentiment_distribution(self, df, save_path=None):
        """
        Plot the distribution of positive vs negative sentiments.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with sentiment analysis results
        save_path : str
            Path to save the plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Count plot
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c']  # Green for positive, red for negative

        ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Sentiment Distribution (Count)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sentiment', fontsize=12)
        ax1.set_ylabel('Number of Reviews', fontsize=12)

        # Add count labels on bars
        for i, v in enumerate(sentiment_counts.values):
            ax1.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')

        # Pie chart
        ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90, explode=(0.05, 0.05),
                textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2.set_title('Sentiment Distribution (Percentage)', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sentiment distribution plot saved to {save_path}")

        plt.show()

    def plot_sentiment_by_rating(self, df, save_path=None):
        """
        Plot sentiment distribution across different ratings.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with sentiment analysis results
        save_path : str
            Path to save the plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Average polarity by rating
        avg_polarity = df.groupby('rating')['polarity'].mean().sort_index()
        colors_polarity = ['#e74c3c' if x < 0 else '#2ecc71' for x in avg_polarity.values]

        ax1.bar(avg_polarity.index, avg_polarity.values, color=colors_polarity, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax1.set_title('Average Sentiment Polarity by Rating', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Rating (Stars)', fontsize=12)
        ax1.set_ylabel('Average Polarity', fontsize=12)
        ax1.set_xticks(avg_polarity.index)

        # Add value labels
        for i, (rating, polarity) in enumerate(avg_polarity.items()):
            ax1.text(rating, polarity + (0.02 if polarity > 0 else -0.02),
                    f'{polarity:.3f}', ha='center', va='bottom' if polarity > 0 else 'top',
                    fontweight='bold')

        # Sentiment count by rating
        sentiment_by_rating = pd.crosstab(df['rating'], df['sentiment'])

        sentiment_by_rating.plot(kind='bar', stacked=False, ax=ax2,
                                 color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
        ax2.set_title('Sentiment Count by Rating', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Rating (Stars)', fontsize=12)
        ax2.set_ylabel('Number of Reviews', fontsize=12)
        ax2.legend(title='Sentiment', fontsize=10)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sentiment by rating plot saved to {save_path}")

        plt.show()

    def plot_polarity_distribution(self, df, save_path=None):
        """
        Plot the distribution of polarity scores.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with sentiment analysis results
        save_path : str
            Path to save the plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Histogram
        ax1.hist(df['polarity'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral (0)')
        ax1.set_title('Distribution of Sentiment Polarity', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Polarity Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()

        # Box plot by sentiment
        sentiment_data = [df[df['sentiment'] == 'Positive']['polarity'],
                         df[df['sentiment'] == 'Negative']['polarity']]

        bp = ax2.boxplot(sentiment_data, labels=['Positive', 'Negative'],
                        patch_artist=True, showmeans=True)

        # Color the boxes
        colors = ['#2ecc71', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_title('Polarity Distribution by Sentiment', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Polarity Score', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Polarity distribution plot saved to {save_path}")

        plt.show()

    def plot_review_length_analysis(self, df, save_path=None):
        """
        Analyze and plot review length vs sentiment.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with sentiment analysis results
        save_path : str
            Path to save the plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot: Review length vs Polarity
        positive_reviews = df[df['sentiment'] == 'Positive']
        negative_reviews = df[df['sentiment'] == 'Negative']

        ax1.scatter(positive_reviews['word_count'], positive_reviews['polarity'],
                   alpha=0.5, c='#2ecc71', label='Positive', s=30)
        ax1.scatter(negative_reviews['word_count'], negative_reviews['polarity'],
                   alpha=0.5, c='#e74c3c', label='Negative', s=30)

        ax1.set_title('Review Word Count vs Sentiment Polarity', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Word Count', fontsize=12)
        ax1.set_ylabel('Polarity', fontsize=12)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Average word count by sentiment
        avg_length = df.groupby('sentiment')['word_count'].mean()

        ax2.bar(avg_length.index, avg_length.values,
               color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
        ax2.set_title('Average Review Length by Sentiment', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Sentiment', fontsize=12)
        ax2.set_ylabel('Average Word Count', fontsize=12)

        # Add value labels
        for i, (sentiment, length) in enumerate(avg_length.items()):
            ax2.text(i, length + 1, f'{length:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Review length analysis plot saved to {save_path}")

        plt.show()

    def create_wordcloud(self, df, sentiment_type='Positive', save_path=None):
        """
        Create a word cloud for specified sentiment type.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with sentiment analysis results
        sentiment_type : str
            Type of sentiment to create word cloud for ('Positive' or 'Negative')
        save_path : str
            Path to save the plot (optional)
        """
        # Filter reviews by sentiment
        filtered_reviews = df[df['sentiment'] == sentiment_type]['preprocessed_text']

        # Combine all reviews into one text
        text = ' '.join(filtered_reviews.astype(str))

        if not text.strip():
            print(f"No text available for {sentiment_type} sentiment word cloud.")
            return

        # Create word cloud
        wordcloud = WordCloud(width=1200, height=600,
                            background_color='white',
                            colormap='Greens' if sentiment_type == 'Positive' else 'Reds',
                            max_words=100,
                            relative_scaling=0.5,
                            min_font_size=10).generate(text)

        # Plot
        plt.figure(figsize=(14, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {sentiment_type} Reviews',
                 fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout(pad=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{sentiment_type} word cloud saved to {save_path}")

        plt.show()

    def plot_sentiment_categories(self, df, save_path=None):
        """
        Plot detailed sentiment categories distribution.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with sentiment analysis results
        save_path : str
            Path to save the plot (optional)
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        category_counts = df['sentiment_category'].value_counts().sort_index()

        colors = ['#c0392b', '#e74c3c', '#95a5a6', '#2ecc71', '#27ae60']
        category_order = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']

        # Reorder categories
        category_counts = category_counts.reindex([cat for cat in category_order if cat in category_counts.index])

        ax.barh(category_counts.index, category_counts.values,
               color=colors[:len(category_counts)], alpha=0.7, edgecolor='black')

        ax.set_title('Detailed Sentiment Category Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Reviews', fontsize=12)
        ax.set_ylabel('Sentiment Category', fontsize=12)

        # Add value labels
        for i, v in enumerate(category_counts.values):
            ax.text(v + 2, i, str(v), va='center', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sentiment categories plot saved to {save_path}")

        plt.show()

    def create_comprehensive_dashboard(self, df, output_dir='data/visualizations'):
        """
        Create all visualizations and save them.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with sentiment analysis results
        output_dir : str
            Directory to save all plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("Creating comprehensive visualization dashboard...")

        # 1. Sentiment Distribution
        print("\n1. Creating sentiment distribution plot...")
        self.plot_sentiment_distribution(df, f'{output_dir}/sentiment_distribution.png')

        # 2. Sentiment by Rating
        print("\n2. Creating sentiment by rating plot...")
        self.plot_sentiment_by_rating(df, f'{output_dir}/sentiment_by_rating.png')

        # 3. Polarity Distribution
        print("\n3. Creating polarity distribution plot...")
        self.plot_polarity_distribution(df, f'{output_dir}/polarity_distribution.png')

        # 4. Review Length Analysis
        print("\n4. Creating review length analysis plot...")
        self.plot_review_length_analysis(df, f'{output_dir}/review_length_analysis.png')

        # 5. Positive Word Cloud
        print("\n5. Creating positive reviews word cloud...")
        self.create_wordcloud(df, 'Positive', f'{output_dir}/wordcloud_positive.png')

        # 6. Negative Word Cloud
        print("\n6. Creating negative reviews word cloud...")
        self.create_wordcloud(df, 'Negative', f'{output_dir}/wordcloud_negative.png')

        # 7. Sentiment Categories
        print("\n7. Creating sentiment categories plot...")
        self.plot_sentiment_categories(df, f'{output_dir}/sentiment_categories.png')

        print(f"\nAll visualizations saved to {output_dir}/")


# Example usage
if __name__ == "__main__":
    # Load sentiment analysis results
    df = pd.read_csv('data/sentiment_analysis_results.csv')

    # Initialize visualizer
    visualizer = SentimentVisualizer()

    # Create comprehensive dashboard
    visualizer.create_comprehensive_dashboard(df)
