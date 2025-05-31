import requests
from bs4 import BeautifulSoup
import json

BASE_URL = "https://peraturan.bpk.go.id"

def get_peraturan_list(page=1):
    url = f"https://peraturan.bpk.go.id/Search?keywords=pembangunan+fasilitas+kesehatan&tentang=&nomor=&p={page}"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.select("table.table tbody tr")
    peraturan_list = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 5:
            continue
        nomor = cols[0].get_text(strip=True)
        tahun = cols[1].get_text(strip=True)
        judul = cols[2].get_text(strip=True)
        jenis = cols[3].get_text(strip=True)
        link_detail = cols[4].find("a")
        detail_url = BASE_URL + link_detail["href"] if link_detail else None
        peraturan_list.append({
            "nomor": nomor,
            "tahun": tahun,
            "judul": judul,
            "jenis": jenis,
            "detail_url": detail_url
        })
    return peraturan_list

# Contoh: Scrape 3 halaman pertama
all_peraturan = []
for page in range(1, 4):
    print(f"Scraping page {page} ...")
    peraturan = get_peraturan_list(page)
    all_peraturan.extend(peraturan)

# Simpan ke file JSON
with open("peraturan_list.json", "w", encoding="utf-8") as f:
    json.dump(all_peraturan, f, ensure_ascii=False, indent=2)

print(f"Total peraturan: {len(all_peraturan)}")
