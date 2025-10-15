import os
import time
import json
import random
import yaml
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
# from urllib.robotparser import RobotFileParser   # commented out (still available if needed)

def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

# def setup_robot_parser(base_url):
#     """Initialize and return a robot parser for the domain."""
#     parsed = urlparse(base_url)
#     robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
#     rp = RobotFileParser()
#     rp.set_url(robots_url)
#     try:
#         rp.read()
#         print(f"[Info] Loaded robots.txt from {robots_url}")
#     except Exception:
#         print(f"[Warning] Could not read robots.txt at {robots_url}")
#     return rp

def scrape_url(url, headers):
    """Fetch HTML content and return text + soup."""
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style"]):  # remove unwanted tags
        tag.decompose()
    text = soup.get_text(separator="\n")
    return text, soup

def is_valid_link(link, base_domain, same_domain_only=True):
    """Validate and normalize link before crawling."""
    if not link or link.startswith("#") or link.startswith("mailto:") or link.startswith("tel:"):
        return False
    parsed = urlparse(link)
    if same_domain_only and parsed.netloc and parsed.netloc != base_domain:
        return False
    return True

def normalize_url(link):
    """Remove fragments and trailing slashes for consistency."""
    link = urldefrag(link)[0]
    if link.endswith("/"):
        link = link[:-1]
    return link

def crawl(url, dest_folder, visited, depth, max_depth, same_domain_only, delay, headers, metadata):
    """Recursively crawl pages up to max_depth."""
    url = normalize_url(url)
    if url in visited:
        return
    if max_depth is not None and max_depth != -1 and depth > max_depth:
        return

    # --- robots.txt check disabled ---
    # if not rp.can_fetch("*", url):
    #     print(f"[Blocked by robots.txt] {url}")
    #     return

    visited.add(url)
    try:
        time.sleep(random.uniform(delay * 0.8, delay * 1.2))  # polite delay
        text, soup = scrape_url(url, headers)
    except Exception as e:
        print(f"[Error] Failed to scrape {url}: {e}")
        return

    os.makedirs(dest_folder, exist_ok=True)
    fname = f"page_{len(visited)}.txt"
    fpath = os.path.join(dest_folder, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(text)

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    metadata.append({
        "url": url,
        "file": fname,
        "title": title,
        "word_count": len(text.split())
    })
    print(f"[Depth {depth}] Saved: {url} -> {fname}")

    base_domain = urlparse(url).netloc
    for a_tag in soup.find_all("a", href=True):
        link = urljoin(url, a_tag["href"])
        link = normalize_url(link)
        if is_valid_link(link, base_domain, same_domain_only):
            crawl(link, dest_folder, visited, depth + 1, max_depth,
                  same_domain_only, delay, headers, metadata)

def save_scrapes(urls, dest_folder, max_depth, same_domain_only, delay, user_agent):
    visited = set()
    metadata = []
    headers = {"User-Agent": user_agent or "ManishRAGBot/1.0"}

    for url in urls:
        # rp = setup_robot_parser(url)  # robots.txt parser disabled
        crawl(url, dest_folder, visited, depth=0, max_depth=max_depth,
              same_domain_only=same_domain_only, delay=delay,
              headers=headers, metadata=metadata)

    # Save metadata
    meta_path = os.path.join(dest_folder, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2, ensure_ascii=False)
    print(f"✅ Finished crawling. {len(visited)} pages saved.")
    print(f"📄 Metadata written to: {meta_path}")

if __name__ == "__main__":
    cfg = load_config()
    urls = cfg["web"]["urls"]
    max_depth = cfg["web"].get("max_depth", -1)
    same_domain_only = cfg["web"].get("same_domain_only", True)
    delay = cfg["web"].get("crawl_delay", 2)
    user_agent = cfg["web"].get("user_agent", "ManishRAGBot/1.0 (contact: you@example.com)")

    save_scrapes(urls, "data_sources/web", max_depth, same_domain_only, delay, user_agent)
