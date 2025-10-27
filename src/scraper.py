"""
Web Scraper Module for Flipkart Product Reviews
This module handles scraping customer reviews from Flipkart product pages
using Selenium and BeautifulSoup.
"""

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


class FlipkartReviewScraper:
    """
    A class to scrape customer reviews from Flipkart product pages.
    """

    def __init__(self, product_url, headless=True):
        """
        Initialize the scraper with product URL and browser settings.

        Parameters:
        -----------
        product_url : str
            The URL of the Flipkart product page
        headless : bool
            Whether to run the browser in headless mode (default: True)
        """
        self.product_url = product_url
        self.headless = headless
        self.driver = None
        self.reviews_data = []

    def setup_driver(self):
        """
        Set up the Selenium WebDriver with Chrome options.
        """
        options = webdriver.ChromeOptions()

        if self.headless:
            options.add_argument('--headless')

        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

        # Use webdriver-manager to automatically download and manage ChromeDriver
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.maximize_window()

    def navigate_to_reviews(self):
        """
        Navigate to the product page and click on reviews section.
        """
        try:
            self.driver.get(self.product_url)
            time.sleep(3)

            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            return True
        except Exception as e:
            print(f"Error navigating to product page: {e}")
            return False

    def extract_reviews_from_page(self):
        """
        Extract reviews from the current page using BeautifulSoup.

        Returns:
        --------
        list : List of dictionaries containing review data
        """
        page_reviews = []

        try:
            # Get page source and parse with BeautifulSoup
            soup = BeautifulSoup(self.driver.page_source, 'lxml')

            # Find all review containers (adjust selectors based on actual Flipkart structure)
            review_containers = soup.find_all('div', class_='col _2wzgFH K0kLPL')

            if not review_containers:
                # Alternative selector patterns
                review_containers = soup.find_all('div', {'class': ['_1AtVbE', 'col-12-12']})

            for review in review_containers:
                try:
                    # Extract username
                    username_elem = review.find('p', class_='_2sc7ZR _2V5EHH')
                    username = username_elem.text.strip() if username_elem else 'Anonymous'

                    # Extract rating
                    rating_elem = review.find('div', class_='_3LWZlK _1BLPMq')
                    rating = None
                    if rating_elem:
                        rating_text = rating_elem.text.strip()
                        rating = float(rating_text) if rating_text.replace('.', '').isdigit() else None

                    # Extract review text
                    review_text_elem = review.find('div', class_='t-ZTKy')
                    if not review_text_elem:
                        review_text_elem = review.find('div', {'class': 'qwjRop'})

                    review_text = review_text_elem.text.strip() if review_text_elem else ''

                    # Only add if we have review text
                    if review_text:
                        page_reviews.append({
                            'username': username,
                            'rating': rating,
                            'review_text': review_text
                        })

                except Exception as e:
                    print(f"Error extracting individual review: {e}")
                    continue

        except Exception as e:
            print(f"Error extracting reviews from page: {e}")

        return page_reviews

    def click_next_page(self):
        """
        Click on the next page button to load more reviews.

        Returns:
        --------
        bool : True if successfully navigated to next page, False otherwise
        """
        try:
            # Try to find and click the "Next" button
            next_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, '_1LKTO3') and contains(text(), 'Next')]"))
            )

            next_button.click()
            time.sleep(3)  # Wait for page to load
            return True

        except (TimeoutException, NoSuchElementException):
            print("No more pages available or next button not found.")
            return False
        except Exception as e:
            print(f"Error clicking next page: {e}")
            return False

    def scrape_reviews(self, target_count=300, max_pages=30):
        """
        Scrape reviews from multiple pages until target count is reached.

        Parameters:
        -----------
        target_count : int
            Target number of reviews to scrape (default: 300)
        max_pages : int
            Maximum number of pages to scrape (default: 30)

        Returns:
        --------
        pd.DataFrame : DataFrame containing all scraped reviews
        """
        self.setup_driver()

        if not self.navigate_to_reviews():
            print("Failed to navigate to product page.")
            self.close()
            return pd.DataFrame()

        print(f"Starting to scrape reviews. Target: {target_count} reviews")

        page_num = 1

        while len(self.reviews_data) < target_count and page_num <= max_pages:
            print(f"Scraping page {page_num}...")

            page_reviews = self.extract_reviews_from_page()

            if page_reviews:
                self.reviews_data.extend(page_reviews)
                print(f"  Extracted {len(page_reviews)} reviews. Total: {len(self.reviews_data)}")
            else:
                print("  No reviews found on this page.")

            # Check if we've reached target
            if len(self.reviews_data) >= target_count:
                print(f"Target of {target_count} reviews reached!")
                break

            # Try to go to next page
            if not self.click_next_page():
                print("No more pages available.")
                break

            page_num += 1
            time.sleep(2)  # Be respectful to the server

        self.close()

        # Convert to DataFrame and limit to target count
        df = pd.DataFrame(self.reviews_data[:target_count])
        print(f"\nScraping completed. Total reviews collected: {len(df)}")

        return df

    def close(self):
        """
        Close the browser and clean up resources.
        """
        if self.driver:
            self.driver.quit()
            print("Browser closed.")


def save_reviews(df, filepath='data/raw_reviews.csv'):
    """
    Save scraped reviews to a CSV file.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing review data
    filepath : str
        Path to save the CSV file
    """
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Reviews saved to {filepath}")


def load_reviews(filepath='data/raw_reviews.csv'):
    """
    Load reviews from a CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame : DataFrame containing review data
    """
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"Loaded {len(df)} reviews from {filepath}")
    return df


# Example usage
if __name__ == "__main__":
    # Example Flipkart iPhone 15 128GB URL (replace with actual URL)
    PRODUCT_URL = "https://www.flipkart.com/apple-iphone-15-black-128-gb/p/itm6d16e1cf03604"

    # Initialize scraper
    scraper = FlipkartReviewScraper(PRODUCT_URL, headless=False)

    # Scrape reviews
    reviews_df = scraper.scrape_reviews(target_count=300, max_pages=30)

    # Save to CSV
    if not reviews_df.empty:
        save_reviews(reviews_df, 'data/raw_reviews.csv')
        print(f"\nDataFrame shape: {reviews_df.shape}")
        print(f"\nFirst few reviews:\n{reviews_df.head()}")
