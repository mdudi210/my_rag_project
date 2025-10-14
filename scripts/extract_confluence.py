import os
import requests
import base64
from bs4 import BeautifulSoup
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def fetch_confluence_pages(base_url, username, api_token, space_key):
    auth = base64.b64encode(f"{username}:{api_token}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}
    pages = []
    url = f"{base_url}/rest/api/content?spaceKey={space_key}&expand=body.storage"
    while url:
        resp = requests.get(url, headers=headers).json()
        for item in resp.get("results", []):
            pages.append({
                "title": item["title"],
                "html": item["body"]["storage"]["value"]
            })
        url = resp.get("_links", {}).get("next")
    return pages

def html_to_text(html):
    return BeautifulSoup(html, "html.parser").get_text(separator="\n")

def save_pages(pages, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for p in pages:
        fname = p["title"].replace(" ", "_") + ".txt"
        text = html_to_text(p["html"])
        path = os.path.join(dest_folder, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

if __name__ == "__main__":
    cfg = load_config()
    pages = fetch_confluence_pages(
        cfg["confluence"]["base_url"],
        cfg["confluence"]["username"],
        cfg["confluence"]["api_token"],
        cfg["confluence"]["space_key"]
    )
    save_pages(pages, "data_sources/confluence")
    print(f"Saved {len(pages)} confluence pages.")
