import csv
import time
import re
import os
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

class FlipkartScrapper:
    def __init__(self, output_dir='data'):
        '''
        Initialize the FlipkartScrapper class.
        '''
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_top_reviews(self, product_url, count=2):
        '''
        Get the top reviews for a given product URL.
        '''
        options = uc.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-blink-features=AutomationControlled')
        driver = uc.Chrome(options=options, use_subprocess=True)

        if not product_url.startswith('http'):
            return 'No product reviews found.'
        
        try:
            driver.get(product_url)
            time.sleep(4)
            try:
                driver.find_element(By.XPATH, "//button[contains(text(), 'X')]").click()
                time.sleep(2)
            except Exception as e:
                print(f'Error occured while closing popup: {e}')

            for _ in range(4):
                ActionChains(driver).send_keys(Keys.END).perform()
                time.sleep(1.5)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            review_blocks = soup.select('div._27M-vq, div.col.EPCmJX, div._6K-7Co')
            seen = set()
            reviews = []

            for block in review_blocks:
                text = block.get_text(separator=' ', strip=True)
                if text and text not in seen:
                    seen.add(text)
                    reviews.append(text)
                if len(reviews) >= count:
                    break
        except Exception:
            reviews = []

        driver.quit()
        return ' || '.join(reviews) if reviews else 'No product reviews found.'

    def scrape_flipkart_product(self, query, max_products=1, review_count=2):
        '''
        Scrape Flipkart for products matching the query and get their reviews.
        '''
        options = uc.ChromeOptions()
        driver = uc.Chrome(options=options, use_subprocess=True)
        search_url = f'https://www.flipkart.com/search?q={query.replace(" ", "+")}'
        driver.get(search_url)
        time.sleep(4)

        try:
            driver.find_element(By.XPATH, "//button[contains(text(), 'X')]").click()
            time.sleep(2)
        except Exception as e:
            print(f'Error occured while closing popup: {e}')

        time.sleep(2)
        products = []
        items = driver.find_elements(By.CSS_SELECTOR, 'div[data-id]')[:max_products]
        for item in items:
            try:
                title = item.find_element(By.CSS_SELECTOR, 'div.KzDlHZ').text.strip()
                price = item.find_element(By.CSS_SELECTOR, 'div.Nx9bqj').text.strip()
                rating = item.find_element(By.CSS_SELECTOR, 'div.XQDdHH').text.strip()
                reviews_text = item.find_element(By.CSS_SELECTOR, 'span.Whh3N').text.strip()
                match = re.search(r'\d+(,\d+)?(?=\s+Reviews)', reviews_text)
                total_reviews = match.group(0) if match else 'N/A'

                link_el = item.find_element(By.CSS_SELECTOR, 'a[href*=\'/p/\']')
                href = link_el.get_attribute('href')
                product_link = href if href.startswith('http') else f'https://www.flipkart.com{href}' #type: ignore
                match = re.findall(r"/p/(itm[0-9a-zA-Z]+)", href) #type: ignore
                product_id = match[0] if match else 'N/A'
            
            except Exception as e:
                print(f'Error occured while processing an item: {e}')
                continue

            top_reviews = self.get_top_reviews(product_link, count=review_count)
            products.append([product_id, title, rating, total_reviews, price, top_reviews])
        driver.quit()
        return products

    def save_to_csv(self, data, filename='product_reviews.csv'):
        '''
        Save the scraped data to a CSV file.
        '''
        if os.path.isabs(filename):
            path = filename

        elif os.path.dirname(filename):
            path = filename
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        else:
            path = os.path.join(self.output_dir, filename)

        with open(path, "w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['product_id', 'product_title', 'rating', 'total_reviews', 'price', 'top_reviews'])
            writer.writerows(data)
