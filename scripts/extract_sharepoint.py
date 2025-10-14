import os
import requests
import yaml
from io import BytesIO
from zipfile import ZipFile
import mimetypes

# For docx: use python-docx, for pdf: PyMuPDF or pdfminer

from docx import Document
import fitz  # PyMuPDF

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def list_sharepoint_files(access_token, site_id, drive_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root/children"
    resp = requests.get(url, headers=headers).json()
    return resp.get("value", [])

def download_file(item, access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    download_url = item["@microsoft.graph.downloadUrl"]
    resp = requests.get(download_url, headers=headers)
    return resp.content

def extract_text_from_bytes(data: bytes, filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        return text
    elif ext in [".docx", ".doc"]:
        # Write to temp file or load from bytes
        from io import BytesIO
        doc = Document(BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext in [".txt", ".md"]:
        return data.decode("utf-8", errors="ignore")
    else:
        return ""

def save_sharepoint_docs(docs, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for item in docs:
        name = item["name"]
        content = download_file(item, cfg["sharepoint"]["access_token"])
        text = extract_text_from_bytes(content, name)
        path = os.path.join(dest_folder, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

if __name__ == "__main__":
    cfg = load_config()
    items = list_sharepoint_files(
        cfg["sharepoint"]["access_token"],
        cfg["sharepoint"]["site_id"],
        cfg["sharepoint"]["drive_id"]
    )
    save_sharepoint_docs(items, "data_sources/sharepoint")
    print(f"Downloaded & saved {len(items)} docs.")
