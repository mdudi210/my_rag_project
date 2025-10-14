from bs4 import BeautifulSoup
import requests
import os
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def scrape_url(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text(separator="\n")

def save_scrapes(urls, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for i, url in enumerate(urls):
        text = scrape_url(url)
        fname = f"page_{i}.txt"
        path = os.path.join(dest_folder, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

if __name__ == "__main__":
    cfg = load_config()
    save_scrapes(cfg["web"]["urls"], "data_sources/web")
    print("Scraped web pages.")
