import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Set
from tqdm import tqdm
import re
import logging
from fake_useragent import UserAgent
import time


class WebsiteExtractor:
    def __init__(self, base_url: str, max_pages: int = 20, request_timeout: int = 10, delay: float = 0.5):
        """
        Advanced website extractor with robust crawling and content extraction.
        """
        self.base_url = self.normalize_url(base_url)
        self.max_pages = max_pages
        self.request_timeout = request_timeout
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.domain = urlparse(self.base_url).netloc
        self.ua = UserAgent()  # ✅ FIXED: must come before _create_session
        self.session = self._create_session()
        self.logger = self._setup_logger()
        self.blacklisted_extensions = {'.pdf', '.jpg', '.png', '.zip', '.gif', '.mp4', '.mp3'}

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler('crawler.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def _create_session(self):
        session = requests.Session()
        session.headers.update({
            'User-Agent': self.ua.random,
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        return session

    def normalize_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            path = parsed.path.rstrip('/') or '/'
            return f"{parsed.scheme}://{parsed.netloc}{path}"
        except Exception as e:
            self.logger.error(f"URL normalization failed for {url}: {str(e)}")
            return url

    def is_valid_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            if parsed.netloc != self.domain:
                return False
            if url in self.visited_urls:
                return False
            if any(parsed.path.lower().endswith(ext) for ext in self.blacklisted_extensions):
                return False
            blacklist = ['/tag/', '/category/', '/author/', '/feed/', '/wp-json/']
            if any(bl in parsed.path.lower() for bl in blacklist):
                return False
            return True
        except Exception as e:
            self.logger.warning(f"URL validation failed for {url}: {str(e)}")
            return False

    def get_links(self, url: str) -> List[str]:
        try:
            time.sleep(self.delay)
            response = self.session.get(
                url,
                timeout=self.request_timeout,
                allow_redirects=True,
                stream=True
            )
            response.raise_for_status()
            if 'text/html' not in response.headers.get('Content-Type', ''):
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            links = set()

            for element in soup.find_all(['a', 'link', 'area'], href=True):
                href = element['href'].strip()
                if not href or href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                    continue
                absolute_url = urljoin(url, href)
                normalized_url = self.normalize_url(absolute_url)
                if self.is_valid_url(normalized_url):
                    links.add(normalized_url)

            return list(links)

        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Request failed for {url}: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error processing {url}: {str(e)}", exc_info=True)
            return []

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'&\w+;', ' ', text)
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        boilerplate = [
            r'cookie policy', r'privacy policy', r'terms of service',
            r'©\s*\d{4}', r'all rights reserved', r'loading\.{3}',
            r'skip to content', r'back to top', r'read more',
            r'this website uses cookies', r'follow us on'
        ]
        for pattern in boilerplate:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 30]
        text = '\n'.join(lines)
        return text.strip()

    def extract_content(self, url: str) -> Dict[str, str]:
        try:
            time.sleep(self.delay)
            response = self.session.get(
                url,
                timeout=self.request_timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            if 'text/html' not in response.headers.get('Content-Type', ''):
                return {'url': url, 'content': '', 'title': url}

            soup = BeautifulSoup(response.text, 'html.parser')

            for element in soup([
                'script', 'style', 'nav', 'footer', 'iframe',
                'noscript', 'svg', 'button', 'form', 'aside',
                'header', 'figure', 'img', 'ad', 'iframe'
            ]):
                element.decompose()

            content_selectors = [
                {'name': 'main'},
                {'role': 'main'},
                {'class': re.compile(r'content|main|article|post', re.I)},
                {'id': re.compile(r'content|main|article', re.I)},
                {'itemprop': 'articleBody'},
                {'class': 'body'},
                {'class': 'entry-content'}
            ]

            main_content = None
            for selector in content_selectors:
                main_content = soup.find(**selector)
                if main_content:
                    break

            main_content = main_content or soup.find('body') or soup
            text_elements = []
            for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'article']):
                text = element.get_text(' ', strip=True)
                if len(text.split()) > 5:
                    text_elements.append(text)

            text = '\n\n'.join(text_elements)

            title = (
                soup.find('meta', property='og:title') or
                soup.find('meta', attrs={'name': 'title'}) or
                soup.title or
                soup.find('h1') or
                urlparse(url).path.replace('/', ' ').title()
            )
            title_text = title.get('content', '') if hasattr(title, 'get') else (
                title.string if title else url
            )

            return {
                'url': url,
                'content': self.clean_text(text),
                'title': self.clean_text(title_text) or url,
                'status': 'success'
            }

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {url}: {str(e)}")
            return {'url': url, 'content': '', 'title': url, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error processing {url}: {str(e)}", exc_info=True)
            return {'url': url, 'content': '', 'title': url, 'status': 'failed', 'error': str(e)}

    def crawl_website(self) -> List[Dict[str, str]]:
        pages = []
        queue = [self.base_url]
        queue_set = {self.base_url}
        self.visited_urls = set()

        with tqdm(total=self.max_pages, desc="🌐 Crawling website") as pbar:
            while queue and len(pages) < self.max_pages:
                current_url = queue.pop(0)
                queue_set.remove(current_url)

                if current_url in self.visited_urls:
                    continue

                self.visited_urls.add(current_url)
                page_data = self.extract_content(current_url)

                if page_data.get('content'):
                    pages.append(page_data)
                    pbar.update(1)
                    pbar.set_postfix({'Current': current_url[:30] + '...'})

                    new_links = self.get_links(current_url)
                    for link in new_links:
                        if link not in self.visited_urls and link not in queue_set:
                            queue.append(link)
                            queue_set.add(link)

        if pages:
            pages[-1]['crawl_stats'] = {
                'total_pages': len(pages),
                'unique_domains': len({urlparse(p['url']).netloc for p in pages}),
                'success_rate': len([p for p in pages if p.get('status') == 'success']) / len(pages) * 100
            }

        return pages
