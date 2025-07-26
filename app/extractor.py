import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict
from tqdm import tqdm
import re
import logging

class WebsiteExtractor:
    def __init__(self, base_url: str, max_pages: int = 20, request_timeout: int = 10):
        """
        Enhanced website extractor with improved crawling and content extraction.
        """
        self.base_url = self.normalize_url(base_url)
        self.max_pages = max_pages
        self.request_timeout = request_timeout
        self.visited_urls = set()
        self.domain = urlparse(self.base_url).netloc
        self.session = self._create_session()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _create_session(self):
        """Create a requests session with fixed user agent."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/113.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.5'
        })
        return session

    def normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")

    def is_valid_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return (
                parsed.netloc == self.domain and
                parsed.scheme in ('http', 'https') and
                url not in self.visited_urls and
                not any(ext in url.lower() for ext in ['.pdf', '.jpg', '.png', '.zip'])
            )
        except Exception:
            return False

    def get_links(self, url: str) -> List[str]:
        try:
            response = self.session.get(url, timeout=self.request_timeout)
            response.raise_for_status()

            if 'text/html' not in response.headers.get('Content-Type', ''):
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            links = set()

            for element in soup.find_all(['a', 'link'], href=True):
                href = element['href'].strip()
                if not href or href.startswith(('mailto:', 'tel:', 'javascript:')):
                    continue

                absolute_url = urljoin(url, href)
                normalized_url = self.normalize_url(absolute_url)
                if self.is_valid_url(normalized_url):
                    links.add(normalized_url)

            return list(links)

        except Exception as e:
            self.logger.warning(f"Failed to extract links from {url}: {str(e)}")
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
            r'skip to content', r'back to top'
        ]
        for pattern in boilerplate:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text.strip()

    def extract_content(self, url: str) -> Dict[str, str]:
        try:
            response = self.session.get(url, timeout=self.request_timeout)
            response.raise_for_status()

            if 'text/html' not in response.headers.get('Content-Type', ''):
                return {'url': url, 'content': '', 'title': url}

            soup = BeautifulSoup(response.text, 'html.parser')

            for element in soup(['script', 'style', 'nav', 'footer', 
                                 'iframe', 'noscript', 'svg', 'button',
                                 'form', 'aside', 'header']):
                element.decompose()

            content_selectors = [
                {'name': 'main'},
                {'class': 'main-content'},
                {'id': 'content'},
                {'role': 'main'},
                {'itemprop': 'articleBody'}
            ]

            main_content = None
            for selector in content_selectors:
                main_content = soup.find(**selector)
                if main_content:
                    break

            main_content = main_content or soup.find('body') or soup

            text = '\n\n'.join(
                p.get_text(' ', strip=True)
                for p in main_content.find_all(['p', 'h1', 'h2', 'h3', 'li'])
            )

            title = (
                soup.find('meta', property='og:title') or
                soup.find('meta', attrs={'name': 'title'}) or
                soup.title
            )
            title_text = title.get('content', '') if hasattr(title, 'get') else (
                title.string if title else url
            )

            return {
                'url': url,
                'content': self.clean_text(text),
                'title': self.clean_text(title_text) or url
            }

        except Exception as e:
            self.logger.error(f"Failed to extract content from {url}: {str(e)}")
            return {'url': url, 'content': '', 'title': url}

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

                    new_links = self.get_links(current_url)
                    for link in new_links:
                        if link not in self.visited_urls and link not in queue_set:
                            queue.append(link)
                            queue_set.add(link)

        return pages
