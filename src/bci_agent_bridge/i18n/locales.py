"""
Supported locales and language definitions.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class LocaleInfo:
    code: str
    name: str
    native_name: str
    rtl: bool = False
    decimal_separator: str = "."
    thousands_separator: str = ","
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"


# Supported locales for BCI-Agent-Bridge
SUPPORTED_LOCALES: Dict[str, LocaleInfo] = {
    "en": LocaleInfo(
        code="en",
        name="English",
        native_name="English",
        rtl=False,
        decimal_separator=".",
        thousands_separator=",",
        date_format="%Y-%m-%d",
        time_format="%H:%M:%S"
    ),
    "es": LocaleInfo(
        code="es", 
        name="Spanish",
        native_name="Español",
        rtl=False,
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d/%m/%Y",
        time_format="%H:%M:%S"
    ),
    "fr": LocaleInfo(
        code="fr",
        name="French", 
        native_name="Français",
        rtl=False,
        decimal_separator=",",
        thousands_separator=" ",
        date_format="%d/%m/%Y",
        time_format="%H:%M:%S"
    ),
    "de": LocaleInfo(
        code="de",
        name="German",
        native_name="Deutsch", 
        rtl=False,
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d.%m.%Y",
        time_format="%H:%M:%S"
    ),
    "ja": LocaleInfo(
        code="ja",
        name="Japanese",
        native_name="日本語",
        rtl=False,
        decimal_separator=".",
        thousands_separator=",",
        date_format="%Y/%m/%d",
        time_format="%H:%M:%S"
    ),
    "zh": LocaleInfo(
        code="zh",
        name="Chinese (Simplified)",
        native_name="简体中文",
        rtl=False, 
        decimal_separator=".",
        thousands_separator=",",
        date_format="%Y/%m/%d",
        time_format="%H:%M:%S"
    ),
    "pt": LocaleInfo(
        code="pt",
        name="Portuguese",
        native_name="Português",
        rtl=False,
        decimal_separator=",",
        thousands_separator=".",
        date_format="%d/%m/%Y",
        time_format="%H:%M:%S"
    ),
    "ru": LocaleInfo(
        code="ru",
        name="Russian", 
        native_name="Русский",
        rtl=False,
        decimal_separator=",",
        thousands_separator=" ",
        date_format="%d.%m.%Y",
        time_format="%H:%M:%S"
    ),
    "ar": LocaleInfo(
        code="ar",
        name="Arabic",
        native_name="العربية",
        rtl=True,
        decimal_separator=".",
        thousands_separator=",", 
        date_format="%d/%m/%Y",
        time_format="%H:%M:%S"
    ),
    "hi": LocaleInfo(
        code="hi",
        name="Hindi",
        native_name="हिन्दी",
        rtl=False,
        decimal_separator=".",
        thousands_separator=",",
        date_format="%d/%m/%Y",
        time_format="%H:%M:%S"
    )
}

# Default locale
DEFAULT_LOCALE = "en"

# Regional fallbacks
LOCALE_FALLBACKS: Dict[str, str] = {
    "en-US": "en",
    "en-GB": "en", 
    "es-ES": "es",
    "es-MX": "es",
    "fr-FR": "fr",
    "fr-CA": "fr",
    "de-DE": "de",
    "de-AT": "de",
    "ja-JP": "ja",
    "zh-CN": "zh",
    "zh-TW": "zh",
    "pt-BR": "pt",
    "pt-PT": "pt",
    "ru-RU": "ru",
    "ar-SA": "ar",
    "ar-EG": "ar",
    "hi-IN": "hi"
}


def get_locale_info(locale: str) -> LocaleInfo:
    """Get locale information with fallback."""
    # Try exact match first
    if locale in SUPPORTED_LOCALES:
        return SUPPORTED_LOCALES[locale]
    
    # Try fallback
    if locale in LOCALE_FALLBACKS:
        fallback = LOCALE_FALLBACKS[locale]
        return SUPPORTED_LOCALES[fallback]
    
    # Try language code only (e.g., "en" from "en-US")
    lang_code = locale.split("-")[0]
    if lang_code in SUPPORTED_LOCALES:
        return SUPPORTED_LOCALES[lang_code]
    
    # Default fallback
    return SUPPORTED_LOCALES[DEFAULT_LOCALE]


def get_supported_locale_list() -> List[Tuple[str, str, str]]:
    """Get list of supported locales as (code, name, native_name)."""
    return [
        (info.code, info.name, info.native_name)
        for info in SUPPORTED_LOCALES.values()
    ]


def is_rtl_locale(locale: str) -> bool:
    """Check if locale uses right-to-left text direction."""
    locale_info = get_locale_info(locale)
    return locale_info.rtl


def format_number(value: float, locale: str) -> str:
    """Format number according to locale conventions."""
    locale_info = get_locale_info(locale)
    
    # Split into integer and decimal parts
    if "." in str(value):
        integer_part, decimal_part = str(value).split(".")
    else:
        integer_part = str(int(value))
        decimal_part = ""
    
    # Add thousands separators
    if len(integer_part) > 3:
        formatted_integer = ""
        for i, digit in enumerate(reversed(integer_part)):
            if i > 0 and i % 3 == 0:
                formatted_integer = locale_info.thousands_separator + formatted_integer
            formatted_integer = digit + formatted_integer
    else:
        formatted_integer = integer_part
    
    # Combine with decimal part
    if decimal_part:
        return formatted_integer + locale_info.decimal_separator + decimal_part
    else:
        return formatted_integer


def format_date(timestamp: float, locale: str, include_time: bool = False) -> str:
    """Format timestamp according to locale conventions."""
    import datetime
    
    locale_info = get_locale_info(locale)
    dt = datetime.datetime.fromtimestamp(timestamp)
    
    if include_time:
        format_str = f"{locale_info.date_format} {locale_info.time_format}"
    else:
        format_str = locale_info.date_format
    
    return dt.strftime(format_str)


# Medical terminology mappings for different languages
MEDICAL_TERMS: Dict[str, Dict[str, str]] = {
    "en": {
        "emergency": "emergency",
        "pain": "pain", 
        "help": "help",
        "stop": "stop",
        "yes": "yes",
        "no": "no",
        "doctor": "doctor",
        "nurse": "nurse",
        "medication": "medication",
        "therapy": "therapy",
        "session": "session"
    },
    "es": {
        "emergency": "emergencia",
        "pain": "dolor",
        "help": "ayuda", 
        "stop": "alto",
        "yes": "sí",
        "no": "no",
        "doctor": "doctor",
        "nurse": "enfermera",
        "medication": "medicamento",
        "therapy": "terapia",
        "session": "sesión"
    },
    "fr": {
        "emergency": "urgence",
        "pain": "douleur",
        "help": "aide",
        "stop": "arrêt",
        "yes": "oui", 
        "no": "non",
        "doctor": "docteur",
        "nurse": "infirmière",
        "medication": "médicament",
        "therapy": "thérapie",
        "session": "séance"
    },
    "de": {
        "emergency": "Notfall",
        "pain": "Schmerz",
        "help": "Hilfe",
        "stop": "stopp",
        "yes": "ja",
        "no": "nein", 
        "doctor": "Arzt",
        "nurse": "Krankenschwester",
        "medication": "Medikament",
        "therapy": "Therapie",
        "session": "Sitzung"
    },
    "ja": {
        "emergency": "緊急事態",
        "pain": "痛み",
        "help": "助け",
        "stop": "停止",
        "yes": "はい",
        "no": "いいえ",
        "doctor": "医師",
        "nurse": "看護師", 
        "medication": "薬",
        "therapy": "治療",
        "session": "セッション"
    },
    "zh": {
        "emergency": "紧急情况",
        "pain": "疼痛", 
        "help": "帮助",
        "stop": "停止",
        "yes": "是",
        "no": "否",
        "doctor": "医生",
        "nurse": "护士",
        "medication": "药物",
        "therapy": "治疗",
        "session": "会话"
    }
}


def get_medical_term(term: str, locale: str) -> str:
    """Get medical term translation for locale."""
    locale_info = get_locale_info(locale)
    locale_code = locale_info.code
    
    if locale_code in MEDICAL_TERMS and term in MEDICAL_TERMS[locale_code]:
        return MEDICAL_TERMS[locale_code][term]
    
    # Fallback to English
    if term in MEDICAL_TERMS["en"]:
        return MEDICAL_TERMS["en"][term]
    
    return term