"""
=============================================================================
EXTERNAL CSS SCRAPER FOR LOW-CREDIBILITY SOURCES
=============================================================================
Reads saved HTML files from html_low_cred/, finds the first external
<link rel="stylesheet">, downloads it, and saves to css_low_cred/.
Then updates the enriched CSV with ext_css_* features.
=============================================================================
"""

import os
import re
import time
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "html_dir": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\thesis_output\html_low_cred",
    "css_output_dir": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\thesis_output\css_low_cred",
    "enriched_csv": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\thesis_output\low_cred_features_latest.csv",
    "request_delay": 0.5,
    "request_timeout": 15,
}

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# HTTP SESSION
# =============================================================================
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/css,*/*;q=0.1",
})


def find_css_url(html_content, base_url):
    """Find the first external stylesheet URL from an HTML page."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Look for <link rel="stylesheet" href="...">
    for link in soup.find_all("link", rel="stylesheet"):
        href = link.get("href", "")
        if href and not href.startswith("data:"):
            # Resolve relative URLs
            if not href.startswith("http"):
                href = urljoin(base_url, href)
            return href
    
    # Fallback: look for <link> with .css in href
    for link in soup.find_all("link", href=True):
        href = link["href"]
        if ".css" in href.lower() and "icon" not in href.lower():
            if not href.startswith("http"):
                href = urljoin(base_url, href)
            return href
    
    return None


def download_css(url, timeout=15):
    """Download a CSS file. Returns (css_text, status)."""
    try:
        resp = SESSION.get(url, timeout=timeout, allow_redirects=True)
        if resp.status_code != 200:
            return None, f"http_{resp.status_code}"
        
        content = resp.text
        # Verify it's actually CSS, not an HTML error page
        content_start = content.strip()[:200].lower()
        if content_start.startswith("<!doctype") or content_start.startswith("<html") or "<head>" in content_start:
            return None, "html_not_css"
        
        if "{" not in content or content.count("{") < 2:
            return None, "not_css"
        
        return content, "success"
    except requests.Timeout:
        return None, "timeout"
    except Exception as e:
        return None, f"error: {str(e)[:50]}"


def extract_ext_css_features(css_text):
    """Extract the same ext_css_* features as the model comparison script."""
    css_lower = css_text.lower()
    features = {}
    
    features["ext_css_size_bytes"] = len(css_text)
    features["ext_css_num_selectors"] = css_text.count("{")
    features["ext_css_num_classes"] = len(re.findall(r"\.\w+", css_text))
    features["ext_css_num_ids"] = len(re.findall(r"#\w+", css_text))
    features["ext_css_num_media_queries"] = len(re.findall(r"@media", css_lower))
    features["ext_css_has_flexbox"] = int("display:flex" in css_lower.replace(" ", "") or "display: flex" in css_lower)
    features["ext_css_has_grid"] = int("display:grid" in css_lower.replace(" ", "") or "display: grid" in css_lower)
    features["ext_css_has_position_absolute"] = int("position:absolute" in css_lower.replace(" ", ""))
    features["ext_css_uses_universal_selector"] = int(bool(re.search(r"(?<!\w)\*\s*\{", css_text)))
    features["ext_css_uses_web_fonts"] = int(bool(re.search(r"@font-face|@import.*fonts", css_lower)))
    features["ext_css_has_bootstrap"] = int(bool(re.search(r"(container-fluid|col-md-|col-lg-|btn-primary|navbar-)", css_lower)))
    features["ext_css_has_reset"] = int(bool(re.search(r"normalize|reset", css_lower)))
    
    hex_colors = set(re.findall(r"#[0-9a-fA-F]{3,8}", css_text))
    rgb_colors = set(re.findall(r"rgba?\([^)]+\)", css_lower))
    features["ext_css_num_colors"] = len(hex_colors) + len(rgb_colors)
    features["ext_css_num_font_weight_bold"] = len(re.findall(r"font-weight\s*:\s*(bold|[7-9]00)", css_lower))
    features["ext_css_num_important"] = css_lower.count("!important")
    features["ext_css_num_imports"] = len(re.findall(r"@import", css_lower))
    
    font_families = set(re.findall(r"font-family\s*:\s*([^;}{]+)", css_lower))
    features["ext_css_num_font_families"] = len(font_families)
    features["ext_css_num_animations"] = len(re.findall(r"@keyframes|animation\s*:", css_lower))
    features["ext_css_num_transitions"] = len(re.findall(r"transition\s*:", css_lower))
    features["ext_css_num_css_variables"] = len(re.findall(r"--[\w-]+\s*:", css_lower))
    
    selectors = re.findall(r"([^{}]+)\{", css_text)
    if selectors:
        lengths = [len(s.strip()) for s in selectors if s.strip()]
        features["ext_css_avg_selector_length"] = round(np.mean(lengths), 2) if lengths else 0
    else:
        features["ext_css_avg_selector_length"] = 0
    
    return features


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  EXTERNAL CSS SCRAPER FOR LOW-CREDIBILITY SOURCES        ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    os.makedirs(CONFIG["css_output_dir"], exist_ok=True)
    
    # Load saved HTML files
    html_dir = Path(CONFIG["html_dir"])
    html_files = sorted(html_dir.glob("*.html"))
    logger.info(f"Found {len(html_files)} saved HTML files in {html_dir}")
    
    if not html_files:
        logger.error("No HTML files found! Check the path.")
        return
    
    # Process each HTML file
    results = {}
    stats = {"css_found": 0, "css_downloaded": 0, "css_failed": 0, "no_css_link": 0}
    
    for html_path in tqdm(html_files, desc="Scraping external CSS"):
        stem = html_path.stem  # e.g., "breitbart_com"
        domain = stem.replace("_", ".")
        base_url = f"https://{domain}"
        
        try:
            html_content = html_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            stats["css_failed"] += 1
            continue
        
        # Find external CSS URL
        css_url = find_css_url(html_content, base_url)
        
        if not css_url:
            stats["no_css_link"] += 1
            continue
        
        stats["css_found"] += 1
        
        # Download CSS
        css_text, status = download_css(css_url)
        
        if css_text:
            stats["css_downloaded"] += 1
            
            # Save CSS file
            css_path = os.path.join(CONFIG["css_output_dir"], f"{stem}.css")
            with open(css_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(css_text)
            
            # Extract features
            features = extract_ext_css_features(css_text)
            results[stem] = features
        else:
            stats["css_failed"] += 1
        
        time.sleep(CONFIG["request_delay"])
    
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"CSS SCRAPING RESULTS")
    logger.info(f"{'='*60}")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  Total with ext_css features: {len(results)}")
    
    # Update the enriched CSV
    if results:
        logger.info(f"\nUpdating enriched CSV with ext_css features...")
        df = pd.read_csv(CONFIG["enriched_csv"])
        
        # Create a mapping from URL to features
        for idx, row in df.iterrows():
            url = row.get("url", "")
            if pd.notna(url) and url:
                try:
                    domain_key = urlparse(str(url)).netloc.replace(".", "_")
                except:
                    domain_key = ""
                
                if domain_key in results:
                    for feat_name, feat_val in results[domain_key].items():
                        df.at[idx, feat_name] = feat_val
        
        # Fill NaN ext_css columns with 0
        ext_cols = [c for c in df.columns if c.startswith("ext_css_")]
        for col in ext_cols:
            df[col] = df[col].fillna(0)
        
        # Save updated CSV
        df.to_csv(CONFIG["enriched_csv"], index=False)
        logger.info(f"Updated CSV saved with {len(ext_cols)} ext_css columns")
        logger.info(f"Sources with ext_css data: {len(results)}/{len(df)}")
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()