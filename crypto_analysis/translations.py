import json
import os
import re
from glob import glob
from typing import Dict, Tuple

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
AR_FILE = os.path.join(CACHE_DIR, 'descriptions_ar.json')


def _detect_arabic(text: str) -> bool:
    try:
        return re.search(r"[\u0600-\u06FF]", text or "") is not None
    except Exception:
        return False


def _translate_with_googletrans(text: str) -> Tuple[str, str]:
    """Try googletrans; returns (result, engine)."""
    try:
        from googletrans import Translator  # type: ignore
        tr = Translator()
        res = tr.translate(text, dest='ar')
        return res.text, 'googletrans'
    except Exception:
        return "", ''


def _translate_with_deep_translator(text: str) -> Tuple[str, str]:
    """Try deep_translator; returns (result, engine)."""
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        res = GoogleTranslator(source='auto', target='ar').translate(text)
        return res, 'deep_translator'
    except Exception:
        return "", ''


def translate_to_ar(text: str) -> Tuple[str, str]:
    """Translate English text to Arabic using available backends. Returns (translated_text, engine)."""
    if not text:
        return "", ''
    if _detect_arabic(text):
        return text, 'already_ar'

    # Try engines in order
    out, engine = _translate_with_googletrans(text)
    if out:
        return out, engine
    out, engine = _translate_with_deep_translator(text)
    if out:
        return out, engine

    # Fallback: return original text if no engine available
    return text, 'fallback_original'


def load_translations() -> Dict[str, str]:
    """Load Arabic descriptions map from cache/descriptions_ar.json if exists."""
    if os.path.isfile(AR_FILE):
        try:
            with open(AR_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure keys are upper symbols and values strings
                return {str(k).upper(): str(v) for k, v in data.items() if isinstance(v, str)}
        except Exception:
            return {}
    return {}


def build_translations_from_cache() -> Dict[str, object]:
    """Read crypto_details_*.json, translate description to Arabic, and write descriptions_ar.json.

    Returns a summary dict: {count, translated, engine_used}
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    files = sorted(glob(os.path.join(CACHE_DIR, 'crypto_details_*.json')))
    out: Dict[str, str] = load_translations()  # keep existing
    translated = 0
    engines: Dict[str, int] = {}

    for path in files:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
        except Exception:
            continue
        symbol = str(obj.get('symbol', '') or os.path.basename(path).split('_')[-1].split('.')[0]).upper()
        desc = obj.get('description')
        if not isinstance(desc, str) or not desc.strip():
            continue

        # Skip if we already have Arabic text stored
        if symbol in out and _detect_arabic(out[symbol]):
            continue

        ar_text, engine = translate_to_ar(desc)
        # لا نكتب في الملف إلا نصًا عربيًا فعليًا
        if ar_text and ar_text.strip() and _detect_arabic(ar_text):
            out[symbol] = ar_text.strip()
            translated += 1
            engines[engine or 'unknown'] = engines.get(engine or 'unknown', 0) + 1

    # Write file
    with open(AR_FILE, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return {
        'count': len(out),
        'translated': translated,
        'engines': engines,
        'output': AR_FILE,
    }
