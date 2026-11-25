"""Feature definitions, URL heuristic extraction, and scam risk analysis."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

FEATURE_NAMES: list[str] = [
    "UsingIP",
    "LongURL",
    "ShortURL",
    "Symbol@",
    "Redirecting//",
    "PrefixSuffix-",
    "SubDomains",
    "HTTPS",
    "DomainRegLen",
    "Favicon",
    "NonStdPort",
    "HTTPSDomainURL",
    "RequestURL",
    "AnchorURL",
    "LinksInScriptTags",
    "ServerFormHandler",
    "InfoEmail",
    "AbnormalURL",
    "WebsiteForwarding",
    "StatusBarCust",
    "DisableRightClick",
    "UsingPopupWindow",
    "IframeRedirection",
    "AgeofDomain",
    "DNSRecording",
    "WebsiteTraffic",
    "PageRank",
    "GoogleIndex",
    "LinksPointingToPage",
    "StatsReport",
]

URL_FEATURE_NAMES: list[str] = [
    "UsingIP",
    "LongURL",
    "ShortURL",
    "Symbol@",
    "Redirecting//",
    "PrefixSuffix-",
    "SubDomains",
    "HTTPS",
    "NonStdPort",
    "HTTPSDomainURL",
]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "UsingIP": "IP address used instead of domain name (-1: Yes, 1: No)",
    "LongURL": "Length of URL (-1: >=75, 0: 54-74, 1: <54)",
    "ShortURL": "Uses URL shortening service (-1: Yes, 1: No)",
    "Symbol@": "Contains '@' symbol in URL (-1: Yes, 1: No)",
    "Redirecting//": "Contains '//' redirect beyond protocol (-1: Yes, 1: No)",
    "PrefixSuffix-": "Uses '-' prefix/suffix in domain (-1: Yes, 1: No)",
    "SubDomains": "Number of subdomains (-1: >2, 0: 2, 1: <=1)",
    "HTTPS": "Uses HTTPS with trusted certificate (-1: No, 0: Untrusted, 1: Yes)",
    "DomainRegLen": "Domain registration length (-1: <=1 yr, 1: >1 yr)",
    "Favicon": "Favicon loaded from external domain (-1: Yes, 1: No)",
    "NonStdPort": "Uses non-standard port (-1: Yes, 1: No)",
    "HTTPSDomainURL": "Contains 'https' token in domain part (-1: Yes, 1: No)",
    "RequestURL": "External objects loaded from other domains (-1: High, 0: Med, 1: Low)",
    "AnchorURL": "Suspicious anchor tags (#, javascript, external) (-1: High, 0: Med, 1: Low)",
    "LinksInScriptTags": "Links in <meta>, <script>, <link> tags (-1: High, 0: Med, 1: Low)",
    "ServerFormHandler": "Form action points to empty or external domain (-1: Empty, 0: Ext, 1: Same)",
    "InfoEmail": "Submits form info to email via mailto: (-1: Yes, 1: No)",
    "AbnormalURL": "Hostname not matching WHOIS identity (-1: Yes, 1: No)",
    "WebsiteForwarding": "Number of redirects (-1: >=4, 0: 2-3, 1: <=1)",
    "StatusBarCust": "onMouseOver changes status bar (-1: Yes, 1: No)",
    "DisableRightClick": "Right-click disabled (-1: Yes, 1: No)",
    "UsingPopupWindow": "Uses popup windows with text fields (-1: Yes, 1: No)",
    "IframeRedirection": "Uses invisible <iframe> redirection (-1: Yes, 1: No)",
    "AgeofDomain": "Age of domain (-1: <6 months, 1: >=6 months)",
    "DNSRecording": "DNS record exists (-1: No, 1: Yes)",
    "WebsiteTraffic": "Website traffic rank (-1: Low, 0: Med, 1: High)",
    "PageRank": "PageRank score (-1: <0.2, 1: >=0.2)",
    "GoogleIndex": "Indexed by Google (-1: No, 1: Yes)",
    "LinksPointingToPage": "External links pointing to page (-1: 0, 0: 1-2, 1: >2)",
    "StatsReport": "Host/IP in known phishing top lists (-1: Yes, 1: No)",
}

SHORTENING_SERVICES = re.compile(
    r"(bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|"
    r"cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|"
    r"snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|"
    r"fic\.kr|loopt\.us|doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|"
    r"bit\.do|lnkd\.in|db\.tt|qr\.ae|adf\.ly|bitly\.com|cur\.lv|ity\.im|q\.gs|po\.st|"
    r"bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|"
    r"scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net)",
    re.IGNORECASE,
)

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work", "click",
    "link", "zip", "mov", "buzz", "icu", "cam", "rest", "vip", "win", "bid",
}

SCAM_KEYWORDS = {
    "login", "signin", "verify", "verification", "secure", "security", "account",
    "update", "confirm", "banking", "bank", "paypal", "paypa1", "amazon", "arnazon",
    "apple", "app1e", "microsoft", "micros0ft", "google", "g00gle", "netflix",
    "wallet", "crypto", "airdrop", "support", "suspended", "unlock", "prize",
    "winner", "claim", "free", "bonus", "otp", "kyc", "password", "webscr",
}

TRUSTED_BRAND_DOMAINS = {
    "google.com", "microsoft.com", "apple.com", "amazon.com", "paypal.com",
    "netflix.com", "github.com", "wikipedia.org", "linkedin.com", "facebook.com",
}


def _has_ip_address(hostname: str) -> bool:
    if not hostname:
        return False
    host = hostname.split(":")[0]
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return bool(re.match(r"^(0x[0-9a-fA-F]+|\d+)(\.(0x[0-9a-fA-F]+|\d+)){3}$", host))


def extract_url_features(url: str) -> dict[str, int]:
    """Extract URL-observable features and return defaults (1 = benign) for page/WHOIS features."""
    normalized = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", normalized):
        normalized = "http://" + normalized

    parsed = urlparse(normalized)
    hostname = parsed.hostname or ""
    netloc = parsed.netloc or ""

    features: dict[str, int] = {name: 1 for name in FEATURE_NAMES}

    features["UsingIP"] = -1 if _has_ip_address(hostname) else 1

    url_len = len(normalized)
    if url_len < 54:
        features["LongURL"] = 1
    elif url_len <= 75:
        features["LongURL"] = 0
    else:
        features["LongURL"] = -1

    features["ShortURL"] = -1 if SHORTENING_SERVICES.search(normalized) else 1
    features["Symbol@"] = -1 if "@" in normalized else 1

    last_double_slash = normalized.rfind("//")
    features["Redirecting//"] = -1 if last_double_slash > 7 else 1

    features["PrefixSuffix-"] = -1 if "-" in hostname else 1

    clean_host = re.sub(r"^www\.", "", hostname, flags=re.IGNORECASE)
    dot_count = clean_host.count(".")
    if dot_count <= 1:
        features["SubDomains"] = 1
    elif dot_count == 2:
        features["SubDomains"] = 0
    else:
        features["SubDomains"] = -1

    features["HTTPS"] = 1 if parsed.scheme.lower() == "https" else -1
    features["NonStdPort"] = -1 if ":" in netloc and not netloc.endswith((":80", ":443")) else 1
    features["HTTPSDomainURL"] = -1 if "https" in hostname.lower() else 1

    return features


def analyze_url_risk(url: str, ml_phishing_prob: float) -> dict:
    """Combine ML probability with scam-pattern heuristics for reliable URL detection."""
    normalized = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", normalized):
        normalized = "http://" + normalized

    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").lower()
    clean_host = re.sub(r"^www\.", "", hostname)
    full_lower = normalized.lower()
    features = extract_url_features(normalized)

    red_flags: list[str] = []
    safe_signals: list[str] = []
    heuristic_score = 0.0

    # High-risk structural signals
    if features["UsingIP"] == -1:
        heuristic_score += 0.45
        red_flags.append("Uses raw IP address instead of a domain name")
    if features["ShortURL"] == -1:
        heuristic_score += 0.35
        red_flags.append("Uses a URL shortening service that hides the true destination")
    if features["Symbol@"] == -1:
        heuristic_score += 0.35
        red_flags.append("Contains '@' symbol used to mask the real host")
    if features["Redirecting//"] == -1:
        heuristic_score += 0.25
        red_flags.append("Contains extra '//' redirect in URL path")
    if features["HTTPSDomainURL"] == -1:
        heuristic_score += 0.35
        red_flags.append("Spoofs 'https' inside the domain name itself")
    if features["NonStdPort"] == -1:
        heuristic_score += 0.20
        red_flags.append("Uses a non-standard network port")
    if features["HTTPS"] == -1:
        heuristic_score += 0.18
        red_flags.append("No HTTPS encryption (unencrypted HTTP)")
    else:
        safe_signals.append("Uses HTTPS protocol")

    if features["PrefixSuffix-"] == -1:
        heuristic_score += 0.18
        red_flags.append("Contains hyphens '-' in domain (common in lookalike domains)")
    if features["SubDomains"] == -1:
        heuristic_score += 0.18
        red_flags.append("Excessive subdomains detected")
    if features["LongURL"] == -1:
        heuristic_score += 0.12
        red_flags.append("Unusually long URL (>= 75 characters)")

    # TLD & Punycode checks
    tld = clean_host.rsplit(".", 1)[-1] if "." in clean_host else ""
    if tld in SUSPICIOUS_TLDS:
        heuristic_score += 0.30
        red_flags.append(f"Uses high-risk top-level domain (.{tld})")
    if "xn--" in clean_host:
        heuristic_score += 0.35
        red_flags.append("Uses punycode (xn--) which can hide homoglyph spoofing")

    # Scam & brand-impersonation keyword checks
    tokens = set(re.findall(r"[a-z0-9]+", full_lower))
    matched_keywords = sorted(tokens & SCAM_KEYWORDS)
    is_trusted_root = any(
        clean_host == trusted or clean_host.endswith("." + trusted)
        for trusted in TRUSTED_BRAND_DOMAINS
    )

    if matched_keywords and not is_trusted_root:
        kw_boost = min(0.40, 0.15 * len(matched_keywords))
        heuristic_score += kw_boost
        red_flags.append(
            f"Contains scam/credential lure keywords: {', '.join(matched_keywords[:5])}"
        )

    if is_trusted_root and not red_flags:
        safe_signals.append(f"Matches verified domain ({clean_host})")

    if not red_flags:
        safe_signals.append("No suspicious URL obfuscation or scam keywords found")

    heuristic_prob = min(0.99, heuristic_score)
    combined_phishing_prob = max(ml_phishing_prob, heuristic_prob)
    if ml_phishing_prob >= 0.5 and heuristic_prob >= 0.3:
        combined_phishing_prob = min(0.995, (ml_phishing_prob + heuristic_prob) / 1.4)

    if is_trusted_root and features["UsingIP"] == 1 and features["Symbol@"] == 1:
        combined_phishing_prob = min(combined_phishing_prob, 0.10)

    is_phishing = combined_phishing_prob >= 0.50
    return {
        "is_phishing": is_phishing,
        "risk_score": round(combined_phishing_prob * 100, 2),
        "ml_phishing_prob": round(ml_phishing_prob * 100, 2),
        "red_flags": red_flags,
        "safe_signals": safe_signals,
        "features": features,
    }
