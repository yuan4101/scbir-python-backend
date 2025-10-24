import asyncio
import os
import re
import csv
import json
import hashlib
import io
from pathlib import Path
from datetime import datetime

import aiohttp
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


# =========================
# Utilidades - Cookies
# =========================

def load_cookies_from_file(filepath="cookies.json"):
    """Carga cookies desde archivo JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cookies = json.load(f)
        print(f"✅ Cookies cargadas desde {filepath}")
        return cookies
    except FileNotFoundError:
        print(f"❌ Archivo {filepath} no encontrado")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error al parsear JSON: {e}")
        return []


def normalize_cookies(raw, default_domain=".tucarro.com.co"):
    """Normaliza cookies al formato esperado por Playwright."""
    normalized = []
    for c in raw:
        n = {
            "name": c["name"],
            "value": c["value"],
            "domain": c.get("domain", default_domain),
            "path": c.get("path", "/"),
        }
        
        exp = c.get("expirationDate")
        if isinstance(exp, (int, float)):
            n["expires"] = int(exp)
        
        if "httpOnly" in c:
            n["httpOnly"] = bool(c["httpOnly"])
        if "secure" in c:
            n["secure"] = bool(c["secure"])
        
        samesite = c.get("sameSite")
        if samesite in ("Lax", "None", "Strict"):
            n["sameSite"] = samesite
        
        normalized.append(n)
    return normalized


# =========================
# Scraper Principal
# =========================

class TuCarroScraper:
    """Scraper asíncrono para extracción de datos de vehículos."""
    
    def __init__(self, cookies_file="cookies.json"):
        self.headless = os.getenv('HEADLESS', 'True') == 'True'
        self.timeout = int(os.getenv('TIMEOUT', 60000))
        self.images_path = Path(os.getenv('IMAGES_PATH', './images'))
        self.data_path = Path('./data')
        self.image_base_url = os.getenv(
            'IMAGE_PATH',
            'https://oielbczfjirunzydccod.supabase.co/storage/v1/object/public/SCBIR/Carros/'
        )
        self.cookies_file = cookies_file
        
        self.images_path.mkdir(exist_ok=True)
        self.data_path.mkdir(exist_ok=True)
    
    async def setup_browser(self, playwright):
        """Configura navegador con cookies y anti-detección."""
        browser = await playwright.chromium.launch(
            headless=self.headless,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            locale='es-CO',
        )
        
        raw_cookies = load_cookies_from_file(self.cookies_file)
        cookies = normalize_cookies(raw_cookies)
        await context.add_cookies(cookies)
        
        page = await context.new_page()
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        return browser, page
    
    async def download_image(self, image_url, unique_id=None):
        """Descarga imagen y genera nombre único basado en ID."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        try:
                            img = Image.open(io.BytesIO(content))
                            format_lower = img.format.lower() if img.format else 'jpg'
                            ext = f'.{format_lower}'
                            
                            if unique_id:
                                filename = f"{unique_id}{ext}"
                            else:
                                filename = f"{hashlib.md5(image_url.encode()).hexdigest()}{ext}"
                            
                            filepath = self.images_path / filename
                            with open(filepath, 'wb') as f:
                                f.write(content)
                            
                            print(f"✅ Imagen guardada: {filename}")
                            return str(filepath), filename
                        
                        except Exception as e:
                            print(f"❌ Error procesando imagen: {e}")
                            return None, None
                    
                    return None, None
        
        except Exception as e:
            print(f"❌ Error descargando: {e}")
            return None, None
    
    async def extract_data(self, page, url):
        """Extrae datos estructurados de un vehículo."""
        html = await page.content()
        soup = BeautifulSoup(html, 'html.parser')
        
        url_id_match = re.search(r'MCO-(\d+)', url)
        url_id = url_id_match.group(1) if url_id_match else None
        
        data = {
            'url': url,
            'marca': None,
            'modelo': None,
            'año': None,
            'precio': None,
            'imagen': None,
        }
        
        price = soup.find('span', class_='andes-money-amount__fraction')
        if price:
            price_text = price.get_text(strip=True)
            price_numeric = price_text.replace('.', '').replace(',', '')
            try:
                data['precio'] = int(price_numeric)
            except ValueError:
                data['precio'] = None
        
        specs = soup.find_all('tr', class_='andes-table__row')
        for spec in specs:
            header = spec.find('th')
            value = spec.find('td')
            if header and value:
                header_text = header.get_text(strip=True).lower()
                value_text = value.get_text(strip=True)
                
                if 'marca' in header_text:
                    data['marca'] = value_text
                elif 'modelo' in header_text:
                    data['modelo'] = value_text
                elif 'año' in header_text:
                    data['año'] = value_text
        
        main_img = soup.find('img', class_='ui-pdp-image ui-pdp-gallery__figure__image')
        if not main_img:
            main_img = soup.find('img', class_='ui-pdp-image')
        if not main_img:
            main_img = soup.find('figure', class_='ui-pdp-gallery__figure')
            if main_img:
                main_img = main_img.find('img')
        
        if main_img:
            img_url = main_img.get('src') or main_img.get('data-src') or main_img.get('data-zoom')
            if img_url:
                local_path, filename = await self.download_image(img_url, url_id)
                if filename:
                    data['imagen'] = f"{self.image_base_url}{filename}"
        
        return data
    
    async def scrape_url(self, page, url, retry=0, max_retries=2):
        """Scraper una URL con sistema de reintentos."""
        clean_url = url.split('#')[0].split('?')[0].strip()
        print(f"\n🔄 [{retry + 1}/{max_retries + 1}] Procesando: {clean_url}")
        
        try:
            await page.goto(clean_url, wait_until='domcontentloaded', timeout=self.timeout)
            wait_time = 5 if retry == 0 else 3
            await asyncio.sleep(wait_time)
            
            data = await self.extract_data(page, clean_url)
            
            if data['precio'] is None and retry < max_retries:
                print("⚠️  Datos incompletos, reintentando...")
                await asyncio.sleep(3)
                return await self.scrape_url(page, clean_url, retry + 1, max_retries)
            
            marca = data['marca'] or 'N/A'
            modelo = data['modelo'] or 'N/A'
            precio = f"${data['precio']:,}" if data['precio'] else 'N/A'
            print(f"✅ {marca} {modelo} - {precio}")
            
            return data
        
        except Exception as e:
            print(f"❌ Error: {e}")
            if retry < max_retries:
                print("🔄 Reintentando...")
                await asyncio.sleep(6)
                return await self.scrape_url(page, clean_url, retry + 1, max_retries)
            return None
    
    async def scrape_multiple(self, urls):
        """Procesa múltiples URLs secuencialmente."""
        print(f"🚀 Iniciando scraping de {len(urls)} vehículos\n")
        results = []
        
        async with async_playwright() as p:
            browser, page = await self.setup_browser(p)
            
            try:
                for idx, url in enumerate(urls, 1):
                    print(f"\n{'='*60}")
                    print(f"Vehículo {idx}/{len(urls)}")
                    print(f"{'='*60}")
                    
                    data = await self.scrape_url(page, url)
                    if data:
                        results.append(data)
                    
                    await asyncio.sleep(2)
            
            finally:
                await browser.close()
        
        self.save_results(results)
        return results
    
    def save_results(self, results):
        """Guarda resultados en CSV y JSON."""
        if not results:
            print("\n⚠️  No hay resultados para guardar")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        csv_file = self.data_path / f'vehiculos_{timestamp}.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n💾 CSV guardado: {csv_file}")
        
        json_file = self.data_path / f'vehiculos_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON guardado: {json_file}")


# =========================
# Punto de Entrada
# =========================

async def main():
    """Carga URLs y ejecuta el scraper."""
    with open('urls.txt', 'r', encoding='utf-8') as f:
        urls = []
        for line in f:
            url = line.strip()
            if url and not url.startswith('#'):
                clean_url = url.split('#')[0].split('?')[0].strip()
                urls.append(clean_url)
    
    if not urls:
        print("❌ No se encontraron URLs en urls.txt")
        return
    
    scraper = TuCarroScraper(cookies_file="cookies.json")
    results = await scraper.scrape_multiple(urls)
    
    print(f"\n{'='*60}")
    print(f"✅ Scraping completado: {len(results)}/{len(urls)} vehículos procesados")
    print(f"{'='*60}")


if __name__ == '__main__':
    asyncio.run(main())
