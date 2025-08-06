"""
Internationalization (i18n) support for BCI-Agent-Bridge.
"""

from .translator import TranslationManager, get_translator, _
from .locales import SUPPORTED_LOCALES, DEFAULT_LOCALE
from .neural_commands import NeuralCommandTranslator

__all__ = ["TranslationManager", "get_translator", "_", "SUPPORTED_LOCALES", "DEFAULT_LOCALE", "NeuralCommandTranslator"]