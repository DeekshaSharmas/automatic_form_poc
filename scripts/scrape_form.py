from scripts.saveurl_script import get_saved_url  # Make sure this is correct
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def scrape_saved_url():
    url = get_saved_url()
    if not url:
        return False, "❌ No saved URL found."

    # Setup headless Chrome options
    options = Options()
    options.add_argument('--headless')  # Run in background
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')

    try:
        # Initialize the WebDriver
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), 
            options=options
        )

        driver.get(url)

        # Wait for the form to appear
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'form'))
        )
        print("✅ Form loaded successfully!")

        # Grab the HTML content of the page after it's fully loaded
        html = driver.page_source

        # Save the HTML to a file
        with open('scraped_page.html', 'w', encoding='utf-8') as f:
            f.write(html)

        print("✅ HTML content has been saved to 'scraped_page.html'")

        driver.quit()
        return True, "✅ HTML content scraped and saved to 'scraped_page.html'"

    except Exception as e:
        print("❌ Error:", e)
        return False, f"❌ Scraping failed: {str(e)}"
