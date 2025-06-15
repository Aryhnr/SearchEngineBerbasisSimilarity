import requests
from bs4 import BeautifulSoup
import csv
import time
import json

BASE_URL = 'https://www.antaranews.com'
CATEGORY_PATH = '/olahraga'
MAX_PAGES = 100  

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
}

def get_article_links_from_page(page_url):
    response = requests.get(page_url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.select('div.card__post.card__post-list.card__post__transition.mt-30')

    links = []
    for article in articles:
        a_tag = article.select_one('div.col-md-5 a')
        if a_tag and a_tag.get('href', '').startswith('http'):
            links.append(a_tag['href'])
    return links

def scrape_article_detail(url):
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    # Judul
    title_tag = soup.select_one('div.wrap__article-detail-title h1')
    title = title_tag.text.strip() if title_tag else ''

    # Kategori
    kategori_tag = soup.select('ul.breadcrumbs li.breadcrumbs__item a')
    kategori = kategori_tag[-2].text.strip() if len(kategori_tag) >= 2 else ''

    # Tanggal
    tanggal_tag = soup.select_one('ul.list-inline li.list-inline-item span.text-secondary')
    tanggal = tanggal_tag.text.strip() if tanggal_tag else ''

    # Konten
    content_div = soup.find('div', class_='wrap__article-detail-content')
    content_paragraphs = content_div.find_all(['p', 'blockquote']) if content_div else []

    isi_berita = '\n\n'.join([
        tag.get_text(strip=True) for tag in content_paragraphs
        if not tag.find('script') and not tag.find('span', class_='baca-juga') and 'adsbygoogle' not in tag.text
    ])

    # Gambar
    img_tag = soup.select_one('picture img')
    gambar_url = img_tag['src'] if img_tag and 'src' in img_tag.attrs else ''

    # Penulis
    penulis = ''
    p = soup.find('p', string=lambda t: t and 'Pewarta:' in t) or \
        soup.find('p', class_='text-muted mt-2 small')
    if p:
        for line in p.get_text(separator="\n").strip().split('\n'):
            if 'Pewarta:' in line:
                penulis = line.replace('Pewarta:', '').strip()

    return {
        'judul': title,
        'kategori': kategori,
        'tanggal': tanggal,
        'konten': isi_berita,
        'penulis': penulis,
        'gambar': gambar_url,
        'link': url,
        'sumber': BASE_URL
    }

def scrape_all_articles():
    all_articles = []
    for page_num in range(1, MAX_PAGES + 1):
        page_url = f"{BASE_URL}{CATEGORY_PATH}" if page_num == 1 else f"{BASE_URL}{CATEGORY_PATH}/{page_num}"
        print(f"\n Scraping halaman: {page_url}")
        
        try:
            links = get_article_links_from_page(page_url)
            print(f" {len(links)} artikel ditemukan")
        except Exception as e:
            print(f" Gagal ambil daftar artikel: {e}")
            time.sleep(3)
            continue

        for link in links:
            try:
                print(f" Scraping artikel: {link}")
                article = scrape_article_detail(link)
                all_articles.append(article)
                time.sleep(2)  # Hindari dianggap bot
            except Exception as e:
                print(f" Gagal scraping artikel: {e}")
                time.sleep(2)
    
    return all_articles



if __name__ == '__main__':
    hasil = scrape_all_articles()

    # Simpan ke CSV
    with open('hasil/hasil_berita_olahraga.csv', 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=[
            'judul', 'kategori', 'tanggal', 'konten',
            'penulis', 'gambar', 'link', 'sumber'
        ])
        writer.writeheader()
        writer.writerows(hasil)

    # Simpan ke JSON
    with open('hasil/hasil_berita_olahraga.json', 'w', encoding='utf-8') as f_json:
        json.dump(hasil, f_json, ensure_ascii=False, indent=4)

    print("\n Semua data selesai diambil dan disimpan ke:")
    print("- hasil/hasil_berita_olahraga.csv")
    print("- hasil/hasil_berita_olahraga.json")

