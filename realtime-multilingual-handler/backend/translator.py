# backend/translator.py
"""
Translator helper using deep-translator (good dev/demo choice).
Falls back to identity if translation fails.
"""

from typing import Optional

try:
    from langdetect import detect
except Exception:
    detect = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

def detect_language(text: str) -> str:
    if not text:
        return "en"
    if detect is None:
        return "en"
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "en"

def translate_text(text: str, target: str = "en") -> str:
    """
    Translate `text` to `target` language using deep-translator's GoogleTranslator.
    If anything fails, return the original text (safe fallback).
    """
    if not text:
        return text
    if GoogleTranslator is None:
        return text
    try:
        # Create translator for the target every call (lightweight for demo)
        return GoogleTranslator(source='auto', target=target).translate(text)
    except Exception:
        return text
