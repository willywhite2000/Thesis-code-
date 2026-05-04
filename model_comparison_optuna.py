"""
=============================================================================
MODEL COMPARISON PIPELINE
=============================================================================
Master Thesis - Wilbert | Tilburg University
Supervisor: Dr. Chris Emmery

Compares classification performance across:
  - 3 dataset configs: Baseline, Enriched, Balanced-Enriched
  - 2 algorithms: Random Forest, XGBoost
  - 3 balancing strategies: None, SMOTE, Class Weights

Evaluation: 5-fold stratified CV, macro-averaged F1, McNemar's test.

Usage:
  1. First run enrich_low_credibility.py to generate low_cred_features_latest.csv
  2. Then run: python model_comparison.py
=============================================================================
"""

import json
import re
import os
import warnings
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score, make_scorer
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    # Paths
    "data_dir": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)",
    "html_front_dir": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\html_front",
    "css_front_dir": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\css_front",
    "mbfc_json_path": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\results_expanded.json",
    "enriched_csv": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\thesis_output\low_cred_features_latest.csv",
    "output_dir": r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\thesis_output\model_results_optuna",

    # Experiment settings
    "n_folds": 5,
    "random_state": 42,
}

# =============================================================================
# LOGGING
# =============================================================================
os.makedirs(CONFIG["output_dir"], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(CONFIG["output_dir"], "model_comparison.log"), mode="w"
        ),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE EXTRACTION (for baseline HTML files)
# =============================================================================
from bs4 import BeautifulSoup


class FeatureExtractor:
    """
    Same feature extractor as enrichment script — ensures feature parity
    between baseline and enriched datasets.
    """

    def __init__(self, html, url=""):
        self.html = html
        self.url = url
        self.soup = BeautifulSoup(html, "html.parser")
        self.text = self.soup.get_text(separator=" ", strip=True)

    def extract_all(self):
        features = {}
        features.update(self._structural_features())
        features.update(self._link_features())
        features.update(self._text_features())
        features.update(self._deception_features())
        features.update(self._metadata_features())
        features.update(self._css_features())
        return features

    def _structural_features(self):
        tags = self.soup.find_all(True)
        return {
            "total_tags": len(tags),
            "unique_tags": len(set(t.name for t in tags)),
            "html_length": len(self.html),
            "text_length": len(self.text),
            "text_to_html_ratio": len(self.text) / max(len(self.html), 1),
            "num_divs": len(self.soup.find_all("div")),
            "num_spans": len(self.soup.find_all("span")),
            "num_tables": len(self.soup.find_all("table")),
            "num_forms": len(self.soup.find_all("form")),
            "num_inputs": len(self.soup.find_all("input")),
            "num_images": len(self.soup.find_all("img")),
            "num_videos": len(self.soup.find_all("video")),
            "num_iframes": len(self.soup.find_all("iframe")),
            "num_scripts": len(self.soup.find_all("script")),
            "num_stylesheets": len(self.soup.find_all("link", rel="stylesheet")),
            "num_headings": len(self.soup.find_all(re.compile(r"^h[1-6]$"))),
            "num_paragraphs": len(self.soup.find_all("p")),
            "num_lists": len(self.soup.find_all(["ul", "ol"])),
            "max_dom_depth": self._max_depth(self.soup, 0),
        }

    def _max_depth(self, element, current_depth):
        children = [c for c in element.children if hasattr(c, "children")]
        if not children:
            return current_depth
        return max(self._max_depth(c, current_depth + 1) for c in children[:50])

    def _link_features(self):
        links = self.soup.find_all("a", href=True)
        domain = urlparse(self.url).netloc if self.url else ""
        internal = external = nofollow = empty_href = 0
        for a in links:
            href = a["href"]
            if not href or href in ("#", "javascript:void(0)"):
                empty_href += 1
            elif href.startswith("http"):
                if domain and domain in href:
                    internal += 1
                else:
                    external += 1
            else:
                internal += 1
            if "nofollow" in a.get("rel", []):
                nofollow += 1
        total = max(internal + external, 1)
        return {
            "num_links_total": len(links),
            "num_links_internal": internal,
            "num_links_external": external,
            "num_links_nofollow": nofollow,
            "num_links_empty": empty_href,
            "external_link_ratio": external / total,
            "internal_link_ratio": internal / total,
        }

    def _text_features(self):
        words = self.text.split()
        sentences = [s.strip() for s in re.split(r"[.!?]+", self.text) if s.strip()]
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "avg_sentence_length": np.mean([len(s.split()) for s in sentences]) if sentences else 0,
        }

    def _deception_features(self):
        features = {}
        ad_keywords = [
            "adsense", "doubleclick", "googletag", "adsbygoogle",
            "taboola", "outbrain", "mgid", "revcontent",
            "amazon-adsystem", "adnxs", "pubmatic", "criteo",
        ]
        html_lower = self.html.lower()
        ad_count = sum(html_lower.count(kw) for kw in ad_keywords)
        features["ad_keyword_count"] = ad_count
        features["ad_density"] = ad_count / max(len(self.text.split()), 1)

        iframes = self.soup.find_all("iframe")
        features["ad_iframe_count"] = sum(
            1 for f in iframes
            if any(kw in str(f.get("src", "")).lower() for kw in ad_keywords)
        )

        popup_classes = ["modal", "popup", "overlay", "lightbox", "interstitial"]
        popup_count = 0
        for cls in popup_classes:
            popup_count += len(self.soup.find_all(class_=re.compile(cls, re.I)))
            popup_count += len(self.soup.find_all(id=re.compile(cls, re.I)))
        features["popup_indicator_count"] = popup_count

        headlines = [t.get_text(strip=True) for t in self.soup.find_all(re.compile(r"^h[1-3]$"))]
        features["clickbait_allcaps_headlines"] = sum(1 for h in headlines if h.isupper() and len(h) > 3)
        features["clickbait_exclamation_headlines"] = sum(1 for h in headlines if "!" in h)
        features["clickbait_question_headlines"] = sum(1 for h in headlines if "?" in h)

        features["hidden_element_count"] = (
            len(self.soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)))
            + len(self.soup.find_all(attrs={"hidden": True}))
        )
        features["has_about_page_link"] = int(bool(self.soup.find("a", href=re.compile(r"/about", re.I))))
        features["has_contact_page_link"] = int(bool(self.soup.find("a", href=re.compile(r"/contact", re.I))))
        features["has_privacy_policy_link"] = int(bool(self.soup.find("a", href=re.compile(r"privacy", re.I))))
        features["has_terms_link"] = int(bool(self.soup.find("a", href=re.compile(r"terms", re.I))))

        social_domains = ["facebook.com", "twitter.com", "x.com", "instagram.com",
                          "youtube.com", "tiktok.com", "telegram", "gab.com", "parler"]
        features["social_media_link_count"] = sum(
            1 for a in self.soup.find_all("a", href=True)
            if any(sd in a["href"].lower() for sd in social_domains)
        )

        scripts = self.soup.find_all("script")
        features["inline_script_count"] = sum(1 for s in scripts if s.string)
        features["external_script_count"] = sum(1 for s in scripts if s.get("src"))
        source_domain = urlparse(self.url).netloc if self.url else ""
        script_domains = set()
        for s in scripts:
            src = s.get("src", "")
            if src.startswith("http"):
                sd = urlparse(src).netloc
                if sd and sd != source_domain:
                    script_domains.add(sd)
        features["third_party_script_domains"] = len(script_domains)
        return features

    def _metadata_features(self):
        metas = self.soup.find_all("meta")
        has_desc = has_og = has_tw = has_canon = has_robots = 0
        for m in metas:
            name = m.get("name", "").lower()
            prop = m.get("property", "").lower()
            if name == "description" or prop == "og:description":
                has_desc = 1
            if prop.startswith("og:"):
                has_og = 1
            if name.startswith("twitter:"):
                has_tw = 1
            if name == "robots":
                has_robots = 1
        if self.soup.find("link", rel="canonical"):
            has_canon = 1
        return {
            "meta_tag_count": len(metas),
            "has_meta_description": has_desc,
            "has_og_tags": has_og,
            "has_twitter_tags": has_tw,
            "has_canonical": has_canon,
            "has_robots_meta": has_robots,
        }

    def _css_features(self):
        """
        Extract CSS features from embedded <style> blocks and inline styles.
        Aligned with Welter's CSS feature set (Appendix A, features 27-47).
        """
        # Collect all CSS text from <style> blocks
        css_text = ""
        for style_tag in self.soup.find_all("style"):
            if style_tag.string:
                css_text += style_tag.string + "\n"

        # Also count inline style attributes
        inline_styles = self.soup.find_all(style=True)
        inline_css = " ".join(tag.get("style", "") for tag in inline_styles)

        all_css = css_text + "\n" + inline_css
        css_lower = all_css.lower()

        features = {}

        # --- Size & Complexity ---
        features["css_total_length"] = len(css_text)
        features["num_inline_styles"] = len(inline_styles)

        # Selectors (approximate count from style blocks)
        features["css_num_selectors"] = css_text.count("{")

        # Classes and IDs referenced
        features["css_num_class_selectors"] = len(re.findall(r"\.\w+", css_text))
        features["css_num_id_selectors"] = len(re.findall(r"#\w+", css_text))

        # --- Layout features ---
        features["css_has_flexbox"] = int("display:flex" in css_lower.replace(" ", "") or
                                          "display: flex" in css_lower)
        features["css_has_grid"] = int("display:grid" in css_lower.replace(" ", "") or
                                       "display: grid" in css_lower)
        features["css_has_position_absolute"] = int("position:absolute" in css_lower.replace(" ", "") or
                                                     "position: absolute" in css_lower)
        features["css_has_position_fixed"] = int("position:fixed" in css_lower.replace(" ", "") or
                                                  "position: fixed" in css_lower)

        # --- Media queries (responsive design) ---
        features["css_num_media_queries"] = len(re.findall(r"@media", css_lower))

        # --- Typography ---
        font_families = set(re.findall(r"font-family\s*:\s*([^;}{]+)", css_lower))
        features["css_num_font_families"] = len(font_families)
        features["css_num_font_weight_bold"] = len(re.findall(r"font-weight\s*:\s*(bold|[7-9]00)", css_lower))
        features["css_uses_web_fonts"] = int(bool(re.search(r"@font-face|@import.*fonts", css_lower)))

        # --- Colors ---
        hex_colors = set(re.findall(r"#[0-9a-fA-F]{3,8}", all_css))
        rgb_colors = set(re.findall(r"rgba?\([^)]+\)", css_lower))
        features["css_num_colors"] = len(hex_colors) + len(rgb_colors)

        # --- Animations & Transitions ---
        features["css_num_animations"] = len(re.findall(r"@keyframes|animation\s*:", css_lower))
        features["css_num_transitions"] = len(re.findall(r"transition\s*:", css_lower))

        # --- Code quality indicators ---
        features["css_num_important"] = css_lower.count("!important")
        features["css_num_imports"] = len(re.findall(r"@import", css_lower))
        features["css_uses_universal_selector"] = int(bool(re.search(r"(?<!\w)\*\s*\{", css_text)))
        features["css_num_css_variables"] = len(re.findall(r"--[\w-]+\s*:", css_lower))

        # --- Framework detection ---
        features["css_has_bootstrap"] = int(bool(re.search(
            r"(container-fluid|col-md-|col-lg-|col-sm-|col-xs-|btn-primary|navbar-)", css_lower
        )))
        features["css_has_tailwind"] = int(bool(re.search(
            r"(\.tw-|\.flex\.|\.grid\.)", css_lower
        )))

        # --- CSS reset indicator ---
        features["css_has_reset"] = int(bool(re.search(
            r"(margin\s*:\s*0|padding\s*:\s*0).*\*\s*\{|normalize|reset", css_lower
        )))

        # --- Selector complexity ---
        selectors = re.findall(r"([^{}]+)\{", css_text)
        if selectors:
            lengths = [len(s.strip()) for s in selectors if s.strip()]
            features["css_avg_selector_length"] = np.mean(lengths) if lengths else 0
            features["css_max_selector_length"] = max(lengths) if lengths else 0
        else:
            features["css_avg_selector_length"] = 0
            features["css_max_selector_length"] = 0

        return features


# =============================================================================
# STEP 1: BUILD BASELINE DATASET (from existing html_front + MBFC JSON)
# =============================================================================
def load_mbfc_metadata(path):
    """Load MBFC JSON and create a domain → metadata lookup."""
    logger.info(f"Loading MBFC metadata from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        entries = list(data.values()) if isinstance(data, dict) else data
    except json.JSONDecodeError:
        entries = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Build lookup: domain (from URL or source_url) → entry
    lookup = {}
    for e in entries:
        url = e.get("url", "")
        source_url = e.get("source_url", "")
        for u in [url, source_url]:
            if u and "mediabiasfactcheck" not in u:
                if not u.startswith("http"):
                    u = f"https://{u}"
                domain = urlparse(u).netloc.lower().replace("www.", "")
                if domain:
                    lookup[domain] = e

        # Also try the source slug as domain hint
        source = e.get("source", "")
        if "mediabiasfactcheck.com/" in source:
            slug = source.rstrip("/").split("/")[-1]
            if slug:
                lookup[slug] = e

    logger.info(f"Built metadata lookup for {len(lookup)} domains")
    return entries, lookup


def load_external_css_features(css_dir):
    """
    Load real CSS files from css_front, filtering out HTML error pages.
    Returns a dict: filename_stem -> css features dict.
    """
    css_lookup = {}
    if not os.path.exists(css_dir):
        logger.warning(f"CSS directory not found: {css_dir}")
        return css_lookup
    
    css_files = list(Path(css_dir).glob("*.css"))
    logger.info(f"Scanning {len(css_files)} CSS files...")
    
    real_css = 0
    html_errors = 0
    
    for css_path in css_files:
        try:
            content = css_path.read_text(encoding="utf-8", errors="replace")
            
            # Filter: skip files that are actually HTML error pages
            content_start = content.strip()[:200].lower()
            if content_start.startswith("<!doctype") or content_start.startswith("<html") or "<head>" in content_start:
                html_errors += 1
                continue
            
            # Must have CSS-like content (at least some selectors)
            if "{" not in content or content.count("{") < 3:
                html_errors += 1
                continue
            
            real_css += 1
            css_lower = content.lower()
            stem = css_path.stem.lower()  # e.g., "6abc_com"
            
            features = {}
            features["ext_css_size_bytes"] = len(content)
            features["ext_css_num_selectors"] = content.count("{")
            features["ext_css_num_classes"] = len(re.findall(r"\.\w+", content))
            features["ext_css_num_ids"] = len(re.findall(r"#\w+", content))
            features["ext_css_num_media_queries"] = len(re.findall(r"@media", css_lower))
            features["ext_css_has_flexbox"] = int("display:flex" in css_lower.replace(" ", "") or "display: flex" in css_lower)
            features["ext_css_has_grid"] = int("display:grid" in css_lower.replace(" ", "") or "display: grid" in css_lower)
            features["ext_css_has_position_absolute"] = int("position:absolute" in css_lower.replace(" ", ""))
            features["ext_css_uses_universal_selector"] = int(bool(re.search(r"(?<!\w)\*\s*\{", content)))
            features["ext_css_uses_web_fonts"] = int(bool(re.search(r"@font-face|@import.*fonts", css_lower)))
            features["ext_css_has_bootstrap"] = int(bool(re.search(r"(container-fluid|col-md-|col-lg-|btn-primary|navbar-)", css_lower)))
            features["ext_css_has_reset"] = int(bool(re.search(r"normalize|reset", css_lower)))
            
            hex_colors = set(re.findall(r"#[0-9a-fA-F]{3,8}", content))
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
            
            selectors = re.findall(r"([^{}]+)\{", content)
            if selectors:
                lengths = [len(s.strip()) for s in selectors if s.strip()]
                features["ext_css_avg_selector_length"] = np.mean(lengths) if lengths else 0
            else:
                features["ext_css_avg_selector_length"] = 0
            
            # Map both underscore and dot versions of the filename
            css_lookup[stem] = features
            css_lookup[stem.replace("_", ".")] = features
            
        except Exception:
            continue
    
    logger.info(f"Real CSS files: {real_css}, HTML error pages skipped: {html_errors}")
    return css_lookup


def build_baseline_dataset(html_dir, mbfc_lookup, css_lookup=None):
    """
    Extract features from all HTML files in html_front.
    Match each to MBFC metadata for credibility labels.
    """
    logger.info(f"\nBuilding baseline dataset from: {html_dir}")
    html_files = list(Path(html_dir).glob("*.html"))
    logger.info(f"Found {len(html_files)} HTML files")

    results = []
    skipped = 0

    for html_path in tqdm(html_files, desc="Extracting baseline features"):
        # Derive domain from filename
        # Filenames can be: "bbc.com.html", "reuters.org.html", "breitbart.html"
        filename = html_path.stem  # e.g., "bbc.com" or "breitbart"
        domain_guess = filename.lower().strip()

        # Try multiple matching strategies against MBFC lookup
        meta = None
        
        # 1. Direct match (e.g., "bbc.com" matches domain "bbc.com")
        meta = mbfc_lookup.get(domain_guess)
        
        # 2. Without www
        if not meta:
            meta = mbfc_lookup.get(domain_guess.replace("www.", ""))
        
        # 3. As slug (e.g., filename "breitbart" matches MBFC slug "breitbart")
        if not meta:
            slug = domain_guess.split(".")[0]  # "bbc.com" → "bbc"
            meta = mbfc_lookup.get(slug)
        
        # 4. With common suffixes (e.g., filename "breitbart" → try "breitbart.com")
        if not meta:
            for suffix in [".com", ".org", ".net", ".co.uk", ".news", ".info"]:
                meta = mbfc_lookup.get(f"{domain_guess}{suffix}")
                if meta:
                    break

        cred = ""
        if meta:
            cred = meta.get("mbfc-credibility-rating", "").strip().lower()

        # Skip entries without credibility rating
        if not cred or cred not in ("high credibility", "medium credibility", "low credibility"):
            skipped += 1
            continue

        # Extract features from HTML
        try:
            html_content = html_path.read_text(encoding="utf-8", errors="replace")
            if len(html_content) < 500:
                skipped += 1
                continue

            extractor = FeatureExtractor(html_content, f"https://{domain_guess}")
            features = extractor.extract_all()
            
            # Add MBFC metadata features
            features.update(extract_mbfc_metadata_features(meta))
            
            # Add domain n-gram features
            features.update(extract_domain_ngrams(domain_guess))
            
            # Add external CSS features if available
            if css_lookup:
                css_feats = css_lookup.get(filename.lower()) or css_lookup.get(domain_guess)
                if css_feats:
                    features.update(css_feats)
            
            features["credibility_class"] = cred
            features["source_file"] = filename
            features["dataset"] = "baseline"
            results.append(features)
        except Exception as e:
            logger.warning(f"Failed to extract features from {filename}: {e}")
            skipped += 1

    logger.info(f"Baseline: extracted features from {len(results)} sources ({skipped} skipped)")

    # Class distribution
    dist = Counter(r["credibility_class"] for r in results)
    for cls, count in dist.most_common():
        logger.info(f"  {cls}: {count}")

    return pd.DataFrame(results)


# =============================================================================
# MBFC METADATA FEATURES
# =============================================================================
def extract_mbfc_metadata_features(meta):
    """
    Extract categorical MBFC features: media_type, country, traffic_popularity.
    One-hot encodes the most common categories; rare ones grouped as 'other'.
    """
    features = {}
    
    # --- Media Type ---
    media_type = meta.get("media-type", "").strip().lower() if meta else ""
    common_media = ["newspaper", "website", "tv station", "news agency",
                    "magazine", "organization/foundation", "radio station"]
    for mt in common_media:
        features[f"media_type_{mt.replace('/', '_').replace(' ', '_')}"] = int(media_type == mt)
    features["media_type_other"] = int(media_type not in common_media and media_type != "")
    
    # --- Country ---
    country = meta.get("country", "").strip().lower() if meta else ""
    # Top countries in MBFC dataset
    common_countries = ["usa", "united states", "uk", "united kingdom", "canada",
                        "australia", "india", "germany", "france", "russia",
                        "turkey", "israel", "ireland", "italy", "netherlands"]
    # Normalize
    country_norm = country
    if country_norm in ("united states", "us", "u.s.", "u.s.a."):
        country_norm = "usa"
    if country_norm in ("united kingdom", "great britain", "england"):
        country_norm = "uk"
    
    for c in ["usa", "uk", "canada", "australia", "india", "germany",
              "france", "russia", "turkey", "israel", "ireland", "italy", "netherlands"]:
        features[f"country_{c}"] = int(country_norm == c)
    features["country_other"] = int(country_norm not in ["usa", "uk", "canada", "australia",
                                                          "india", "germany", "france", "russia",
                                                          "turkey", "israel", "ireland", "italy",
                                                          "netherlands"] and country != "")
    
    # --- Traffic Popularity ---
    traffic = meta.get("traffic-popularity", "").strip().lower() if meta else ""
    traffic_levels = ["high traffic", "medium traffic", "minimal traffic"]
    for tl in traffic_levels:
        features[f"traffic_{tl.replace(' ', '_')}"] = int(traffic == tl)
    features["traffic_unknown"] = int(traffic not in traffic_levels)
    
    return features


# =============================================================================
# DOMAIN N-GRAM FEATURES
# =============================================================================
def extract_domain_ngrams(domain, n_range=(2, 4), top_k=50):
    """
    Extract character-level n-grams from a domain name.
    Matches Welter's approach: break URL into character n-grams to capture
    structural patterns (e.g., 'news', 'com', 'www', 'org').
    
    Uses a fixed vocabulary of common discriminative n-grams.
    """
    # Fixed vocabulary of n-grams known to be discriminative from Welter's analysis
    ngram_vocab = [
        # 2-grams
        "ww", "ws", "co", "om", "ne", "ew", "or", "rg", "et", "th",
        "he", "in", "on", "an", "er", "es", "en", "re", "ed", "ti",
        # 3-grams
        "www", "com", "org", "net", "new", "ews", "the", "news", "pos",
        "ost", "dai", "ail", "ily", "tim", "ime", "mes", "rep", "epo",
        "por", "ort", "pol", "oli", "lit", "tic", "ics", "med", "edi",
        "dia", "nat", "ati", "tio", "ion", "nal",
        # 4-grams
        "news", "post", "time", "daily", "media", "report", "polit",
        ".com", ".org", ".net", "wire", "free", "trib", "press",
    ]
    
    # Extract n-grams from domain
    domain_clean = domain.lower().replace("www.", "")
    domain_ngrams = Counter()
    for n in range(2, 5):
        for i in range(len(domain_clean) - n + 1):
            gram = domain_clean[i:i+n]
            if gram in ngram_vocab:
                domain_ngrams[gram] += 1
    
    features = {}
    for gram in ngram_vocab:
        features[f"ngram_{gram}"] = domain_ngrams.get(gram, 0)
    
    return features


# =============================================================================
# STEP 2: BUILD ENRICHED DATASET
# =============================================================================
def build_enriched_dataset(baseline_df, enriched_csv_path, mbfc_entries):
    """Merge baseline with enriched low-credibility sources, adding MBFC metadata + n-grams."""
    logger.info(f"\nLoading enriched low-cred data from: {enriched_csv_path}")

    if not os.path.exists(enriched_csv_path):
        logger.error(f"Enriched CSV not found! Run enrich_low_credibility.py first.")
        return baseline_df

    enriched = pd.read_csv(enriched_csv_path)
    enriched["dataset"] = "enriched"
    logger.info(f"Loaded {len(enriched)} enriched low-credibility sources")

    # Build a quick lookup from MBFC entries for enriched sources
    mbfc_by_source = {}
    for e in mbfc_entries:
        source = e.get("source", "")
        if source:
            mbfc_by_source[source] = e
        # Also index by slug
        if "mediabiasfactcheck.com/" in source:
            slug = source.rstrip("/").split("/")[-1]
            mbfc_by_source[slug] = e

    # Add MBFC metadata + n-gram + CSS features to each enriched row
    # Try to re-extract CSS from saved HTML files
    html_low_cred_dir = os.path.join(os.path.dirname(enriched_csv_path), "html_low_cred")
    saved_html_lookup = {}
    if os.path.exists(html_low_cred_dir):
        for f in Path(html_low_cred_dir).glob("*.html"):
            # Filename is domain with dots replaced by underscores, e.g., "breitbart_com.html"
            saved_html_lookup[f.stem] = f
        logger.info(f"Found {len(saved_html_lookup)} saved HTML files for CSS re-extraction")
    
    css_extracted = 0
    new_feature_rows = []
    for _, row in enriched.iterrows():
        new_features = {}
        
        # Try to find MBFC metadata for this source
        meta = None
        mbfc_url = row.get("mbfc_review_url", "")
        source_name = row.get("source_name", "")
        url = row.get("url", "")
        
        if mbfc_url:
            meta = mbfc_by_source.get(mbfc_url)
            if not meta:
                slug = mbfc_url.rstrip("/").split("/")[-1] if "/" in str(mbfc_url) else ""
                meta = mbfc_by_source.get(slug)
        
        # Extract MBFC metadata features
        new_features.update(extract_mbfc_metadata_features(meta if meta else {}))
        
        # Extract domain n-grams from URL
        domain = ""
        if pd.notna(url) and url:
            try:
                domain = urlparse(str(url)).netloc.lower().replace("www.", "")
            except:
                domain = str(url).lower()
        new_features.update(extract_domain_ngrams(domain))
        
        # Re-extract CSS features from saved HTML if available
        if pd.notna(url) and url and saved_html_lookup:
            try:
                domain_key = urlparse(str(url)).netloc.replace(".", "_")
                html_file = saved_html_lookup.get(domain_key)
                if html_file:
                    html_content = html_file.read_text(encoding="utf-8", errors="replace")
                    temp_extractor = FeatureExtractor(html_content, str(url))
                    css_feats = temp_extractor._css_features()
                    new_features.update(css_feats)
                    css_extracted += 1
            except:
                pass  # CSS features will be 0 for this entry
        
        new_feature_rows.append(new_features)
    
    if css_extracted:
        logger.info(f"Re-extracted CSS features from {css_extracted}/{len(enriched)} saved HTML files")
    
    # Add new features as columns to enriched df
    new_features_df = pd.DataFrame(new_feature_rows)
    enriched = pd.concat([enriched.reset_index(drop=True), new_features_df.reset_index(drop=True)], axis=1)
    logger.info(f"Added MBFC metadata + n-gram + CSS features to enriched data ({len(new_features_df.columns)} new columns)")

    # Ensure feature columns align
    baseline_features = set(baseline_df.columns)
    enriched_features = set(enriched.columns)
    common_features = baseline_features & enriched_features
    logger.info(f"Common features: {len(common_features)}")

    # Combine — keep only common numeric features + labels
    meta_cols = ["credibility_class", "dataset", "source_file", "url",
                 "source_name", "bias_rating", "factual_reporting"]
    
    # Get numeric feature columns present in both
    numeric_baseline = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_enriched = enriched.select_dtypes(include=[np.number]).columns.tolist()
    shared_numeric = sorted(set(numeric_baseline) & set(numeric_enriched))
    
    logger.info(f"Shared numeric features for modelling: {len(shared_numeric)}")

    # Standardise both dataframes
    keep_cols = shared_numeric + [c for c in meta_cols if c in baseline_df.columns or c in enriched.columns]
    
    combined = pd.concat(
        [baseline_df.reindex(columns=keep_cols), enriched.reindex(columns=keep_cols)],
        ignore_index=True,
    )

    # Fill NaN in numeric columns with 0 (feature absence = 0)
    for col in shared_numeric:
        combined[col] = combined[col].fillna(0)

    logger.info(f"\nEnriched dataset total: {len(combined)}")
    dist = Counter(combined["credibility_class"])
    for cls, count in dist.most_common():
        logger.info(f"  {cls}: {count}")

    return combined


# =============================================================================
# STEP 3: MODEL TRAINING & EVALUATION
# =============================================================================
def get_feature_columns(df):
    """Get numeric feature columns (exclude metadata)."""
    exclude = {"credibility_class", "dataset", "source_file", "url",
               "source_name", "bias_rating", "factual_reporting",
               "final_url", "scrape_status", "url_method",
               "extraction_method", "mbfc_review_url"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def run_experiment(df, dataset_name, feature_cols, n_folds=5, random_state=42):
    """
    Run the full experiment grid for one dataset configuration.
    Now includes feature selection and hyperparameter tuning.
    
    Models: RF, XGBoost
    Balancing: None, SMOTE, Class Weights
    
    Returns list of result dicts.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: {dataset_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Samples: {len(df)}, Features: {len(feature_cols)}")

    # Encode labels
    le = LabelEncoder()
    X = df[feature_cols].fillna(0).values
    y = le.fit_transform(df["credibility_class"])
    class_names = le.classes_

    logger.info(f"Classes: {dict(zip(class_names, np.bincount(y)))}")

    # --- Feature Selection: remove low-variance / uninformative features ---
    # Use SelectKBest with ANOVA F-test, keep top features
    n_features_to_keep = min(len(feature_cols), max(80, int(len(feature_cols) * 0.6)))
    selector = SelectKBest(f_classif, k=n_features_to_keep)
    X_selected = selector.fit_transform(X, y)
    selected_mask = selector.get_support()
    selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]
    logger.info(f"Feature selection: {len(feature_cols)} -> {len(selected_features)} features")
    X = X_selected

    # Check minimum class size for SMOTE
    min_class_count = min(np.bincount(y))
    smote_viable = min_class_count >= 6

    # --- Optuna Bayesian Hyperparameter Optimization ---
    skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def rf_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.1, 0.8),
        }
        scores = []
        for train_idx, val_idx in skf_inner.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            model = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(f1_score(y_val, preds, average="macro"))
        return np.mean(scores)

    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        scores = []
        for train_idx, val_idx in skf_inner.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            model = xgb.XGBClassifier(
                **params, random_state=random_state,
                eval_metric="mlogloss", use_label_encoder=False, verbosity=0
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(f1_score(y_val, preds, average="macro"))
        return np.mean(scores)

    # Define model configurations
    configs = []

    # --- Tune RF with Optuna ---
    logger.info(f"\n  Tuning RF with Optuna (50 trials)...")
    rf_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    rf_study.optimize(rf_objective, n_trials=50, show_progress_bar=True)
    best_rf_params = rf_study.best_params
    logger.info(f"  Best RF params: {best_rf_params} (CV score: {rf_study.best_value:.4f})")

    # --- Tune XGBoost with Optuna ---
    logger.info(f"  Tuning XGB with Optuna (50 trials)...")
    xgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)
    best_xgb_params = xgb_study.best_params
    logger.info(f"  Best XGB params: {best_xgb_params} (CV score: {xgb_study.best_value:.4f})")

    # --- No balancing (with tuned params) ---
    configs.append({
        "name": "RF_no_balance",
        "model": RandomForestClassifier(**best_rf_params, random_state=random_state, n_jobs=-1),
        "smote": False,
    })
    configs.append({
        "name": "XGB_no_balance",
        "model": xgb.XGBClassifier(
            **best_xgb_params, random_state=random_state,
            eval_metric="mlogloss", use_label_encoder=False, verbosity=0
        ),
        "smote": False,
    })

    # --- SMOTE (with tuned params) ---
    if smote_viable:
        k = min(5, min_class_count - 1)
        configs.append({
            "name": "RF_SMOTE",
            "model": RandomForestClassifier(**best_rf_params, random_state=random_state, n_jobs=-1),
            "smote": True,
            "smote_k": k,
        })
        configs.append({
            "name": "XGB_SMOTE",
            "model": xgb.XGBClassifier(
                **best_xgb_params, random_state=random_state,
                eval_metric="mlogloss", use_label_encoder=False, verbosity=0
            ),
            "smote": True,
            "smote_k": k,
        })
    else:
        logger.warning(f"SMOTE skipped: min class has only {min_class_count} samples (need >= 6)")

    # --- Class Weights (with tuned params) ---
    class_counts = np.bincount(y)
    total = len(y)
    n_classes = len(class_counts)
    weight_dict = {i: total / (n_classes * class_counts[i]) for i in range(n_classes)}

    rf_weighted_params = {**best_rf_params, "class_weight": "balanced"}
    configs.append({
        "name": "RF_class_weights",
        "model": RandomForestClassifier(**rf_weighted_params, random_state=random_state, n_jobs=-1),
        "smote": False,
    })

    configs.append({
        "name": "XGB_class_weights",
        "model": xgb.XGBClassifier(
            **best_xgb_params, random_state=random_state,
            eval_metric="mlogloss", use_label_encoder=False, verbosity=0
        ),
        "smote": False,
        "sample_weights": weight_dict,
    })

    # --- Run cross-validation ---
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    all_results = []

    for cfg in configs:
        logger.info(f"\n  Running: {cfg['name']}...")

        fold_metrics = []
        fold_predictions = []
        fold_truths = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Apply SMOTE if configured
            if cfg.get("smote"):
                try:
                    smote = SMOTE(k_neighbors=cfg["smote_k"], random_state=random_state)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                except Exception as e:
                    logger.warning(f"    SMOTE failed on fold {fold_idx}: {e}")

            # Train
            model = cfg["model"].__class__(**cfg["model"].get_params())

            if cfg.get("sample_weights"):
                sw = np.array([cfg["sample_weights"][yi] for yi in y_train])
                model.fit(X_train, y_train, sample_weight=sw)
            else:
                model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            fold_predictions.extend(y_pred)
            fold_truths.extend(y_test)

            # Per-fold metrics
            fold_metrics.append({
                "fold": fold_idx,
                "macro_f1": f1_score(y_test, y_pred, average="macro"),
                "accuracy": accuracy_score(y_test, y_pred),
            })

        # Aggregate across folds
        all_preds = np.array(fold_predictions)
        all_truths = np.array(fold_truths)

        # Per-class F1
        per_class_f1 = f1_score(all_truths, all_preds, average=None, labels=range(len(class_names)))
        per_class_precision = precision_score(all_truths, all_preds, average=None, labels=range(len(class_names)))
        per_class_recall = recall_score(all_truths, all_preds, average=None, labels=range(len(class_names)))

        macro_f1_scores = [fm["macro_f1"] for fm in fold_metrics]

        result = {
            "dataset": dataset_name,
            "model": cfg["name"],
            "macro_f1_mean": np.mean(macro_f1_scores),
            "macro_f1_std": np.std(macro_f1_scores),
            "accuracy": accuracy_score(all_truths, all_preds),
        }

        # Add per-class scores
        for i, cls in enumerate(class_names):
            cls_short = cls.replace(" credibility", "").strip()
            result[f"f1_{cls_short}"] = per_class_f1[i]
            result[f"precision_{cls_short}"] = per_class_precision[i]
            result[f"recall_{cls_short}"] = per_class_recall[i]

        all_results.append(result)

        logger.info(f"    Macro F1: {result['macro_f1_mean']:.4f} (±{result['macro_f1_std']:.4f})")
        for i, cls in enumerate(class_names):
            cls_short = cls.replace(" credibility", "").strip()
            logger.info(f"    {cls_short} F1: {per_class_f1[i]:.4f}")

        # Save confusion matrix for this config
        cm = confusion_matrix(all_truths, all_preds)
        _save_confusion_matrix(
            cm, class_names, f"{dataset_name}__{cfg['name']}",
            CONFIG["output_dir"]
        )

    return all_results


def _save_confusion_matrix(cm, class_names, title, output_dir):
    """Save a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    short_names = [c.replace(" credibility", "") for c in class_names]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=short_names, yticklabels=short_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {title}")
    plt.tight_layout()
    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(output_dir, f"cm_{safe_title}.png"), dpi=150)
    plt.close()


# =============================================================================
# STEP 4: MCNEMAR'S TEST (Statistical Significance)
# =============================================================================
def mcnemar_test(y_true, preds_a, preds_b):
    """
    McNemar's test comparing two classifiers.
    Returns chi2 statistic and p-value.
    """
    from scipy.stats import chi2

    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # Contingency table
    b = np.sum(correct_a & ~correct_b)  # A right, B wrong
    c = np.sum(~correct_a & correct_b)  # A wrong, B right

    if b + c == 0:
        return 0.0, 1.0

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return chi2_stat, p_value


# =============================================================================
# STEP 5: VISUALIZATION
# =============================================================================
def plot_comparison_table(results_df, output_dir):
    """Create the main comparison figure for the thesis."""

    # --- Figure 1: Macro F1 by dataset & model ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot = results_df.pivot_table(
        values="macro_f1_mean", index="model", columns="dataset"
    )
    pivot.plot(kind="bar", ax=ax, rot=30, colormap="Set2")
    ax.set_ylabel("Macro-averaged F1")
    ax.set_title("Model Performance Across Dataset Configurations")
    ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 1)

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_macro_f1.png"), dpi=150)
    plt.close()

    # --- Figure 2: Low-credibility F1 specifically ---
    low_col = [c for c in results_df.columns if "f1_low" in c.lower()]
    if low_col:
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_low = results_df.pivot_table(
            values=low_col[0], index="model", columns="dataset"
        )
        pivot_low.plot(kind="bar", ax=ax, rot=30, colormap="Set1")
        ax.set_ylabel("Low-Credibility F1 Score")
        ax.set_title("Low-Credibility Class Performance (Primary Metric)")
        ax.axhline(y=0.533, color="gray", linestyle="--", label="Welter baseline (0.533)")
        ax.axhline(y=0.65, color="green", linestyle="--", alpha=0.5, label="Target (0.65)")
        ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_ylim(0, 1)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_low_cred_f1.png"), dpi=150)
        plt.close()

    logger.info(f"Saved comparison figures to {output_dir}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Model comparison pipeline started at {timestamp}")
    logger.info(f"Config: {json.dumps(CONFIG, indent=2, default=str)}")

    # --- 1. Load MBFC metadata ---
    entries, mbfc_lookup = load_mbfc_metadata(CONFIG["mbfc_json_path"])

    # --- 2. Load external CSS features ---
    css_lookup = load_external_css_features(CONFIG["css_front_dir"])

    # --- 3. Build baseline dataset ---
    baseline_df = build_baseline_dataset(CONFIG["html_front_dir"], mbfc_lookup, css_lookup)

    if baseline_df.empty:
        logger.error("Baseline dataset is empty! Check html_front directory and MBFC matching.")
        return

    feature_cols = get_feature_columns(baseline_df)
    logger.info(f"\nFeature columns ({len(feature_cols)}): {feature_cols[:10]}...")

    # --- 3. Build enriched dataset ---
    enriched_df = build_enriched_dataset(baseline_df, CONFIG["enriched_csv"], entries)

    # Ensure consistent feature columns across all experiments
    enriched_feature_cols = get_feature_columns(enriched_df)
    # Use features present in both
    shared_features = sorted(set(feature_cols) & set(enriched_feature_cols))
    logger.info(f"Shared features for all experiments: {len(shared_features)}")

    # --- 4. Run experiments ---
    all_results = []

    # Experiment 1: Baseline (original data only)
    results_baseline = run_experiment(
        baseline_df, "Baseline", shared_features,
        n_folds=CONFIG["n_folds"], random_state=CONFIG["random_state"]
    )
    all_results.extend(results_baseline)

    # Experiment 2: Enriched (baseline + new low-cred sources)
    results_enriched = run_experiment(
        enriched_df, "Enriched", shared_features,
        n_folds=CONFIG["n_folds"], random_state=CONFIG["random_state"]
    )
    all_results.extend(results_enriched)

    # --- 5. Compile results ---
    results_df = pd.DataFrame(all_results)

    # Save full results table
    results_path = os.path.join(CONFIG["output_dir"], f"results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    results_df.to_csv(os.path.join(CONFIG["output_dir"], "results_latest.csv"), index=False)

    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    
    display_cols = ["dataset", "model", "macro_f1_mean", "macro_f1_std", "accuracy"]
    # Add per-class F1 columns
    for c in results_df.columns:
        if c.startswith("f1_"):
            display_cols.append(c)
    
    summary = results_df[display_cols].round(4)
    logger.info(f"\n{summary.to_string(index=False)}")

    # --- 6. Visualizations ---
    plot_comparison_table(results_df, CONFIG["output_dir"])

    # --- 7. Improvement over baseline ---
    logger.info(f"\n{'='*60}")
    logger.info(f"IMPROVEMENT ANALYSIS")
    logger.info(f"{'='*60}")

    baseline_results = results_df[results_df["dataset"] == "Baseline"]
    enriched_results = results_df[results_df["dataset"] == "Enriched"]

    for model_name in baseline_results["model"].unique():
        b = baseline_results[baseline_results["model"] == model_name].iloc[0]
        e_match = enriched_results[enriched_results["model"] == model_name]
        if not e_match.empty:
            e = e_match.iloc[0]
            delta_macro = e["macro_f1_mean"] - b["macro_f1_mean"]
            logger.info(f"\n  {model_name}:")
            logger.info(f"    Macro F1: {b['macro_f1_mean']:.4f} -> {e['macro_f1_mean']:.4f} (delta = {delta_macro:+.4f})")
            
            # Low-cred specific if available
            low_col = [c for c in results_df.columns if "f1_low" in c.lower()]
            if low_col:
                col = low_col[0]
                delta_low = e[col] - b[col]
                logger.info(f"    Low F1:   {b[col]:.4f} -> {e[col]:.4f} (delta = {delta_low:+.4f})")

    logger.info(f"\nAll results saved to: {CONFIG['output_dir']}")
    logger.info(f"Done!")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  MODEL COMPARISON PIPELINE                               ║
    ║  Baseline vs Enriched × RF/XGBoost × 3 Balancing         ║
    ║  Master Thesis — Tilburg University                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    main()