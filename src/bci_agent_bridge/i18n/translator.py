"""
Translation management system for BCI-Agent-Bridge.
"""

import os
import json
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
import threading

from .locales import SUPPORTED_LOCALES, DEFAULT_LOCALE, get_locale_info


class TranslationManager:
    """Manages translations for BCI-Agent-Bridge."""
    
    def __init__(self, locale: str = DEFAULT_LOCALE, translations_dir: Optional[Path] = None):
        self.current_locale = locale
        self.translations_dir = translations_dir or (Path(__file__).parent / "translations")
        self.translations: Dict[str, Dict[str, str]] = {}
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Load translations
        self._load_translations()
    
    def _load_translations(self) -> None:
        """Load translation files for all supported locales."""
        with self._lock:
            self.translations.clear()
            
            # Ensure translations directory exists
            self.translations_dir.mkdir(exist_ok=True)
            
            for locale_code in SUPPORTED_LOCALES.keys():
                self._load_locale_translations(locale_code)
    
    def _load_locale_translations(self, locale: str) -> None:
        """Load translations for a specific locale."""
        translation_file = self.translations_dir / f"{locale}.json"
        
        try:
            if translation_file.exists():
                with open(translation_file, 'r', encoding='utf-8') as f:
                    translations = json.load(f)
                    self.translations[locale] = translations
                    self.logger.debug(f"Loaded {len(translations)} translations for {locale}")
            else:
                # Create default translation file
                self._create_default_translations(locale)
                
        except Exception as e:
            self.logger.error(f"Failed to load translations for {locale}: {e}")
            self.translations[locale] = {}
    
    def _create_default_translations(self, locale: str) -> None:
        """Create default translation file for a locale."""
        default_translations = self._get_default_translations(locale)
        translation_file = self.translations_dir / f"{locale}.json"
        
        try:
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(default_translations, f, indent=2, ensure_ascii=False)
            
            self.translations[locale] = default_translations
            self.logger.info(f"Created default translations for {locale}")
            
        except Exception as e:
            self.logger.error(f"Failed to create default translations for {locale}: {e}")
            self.translations[locale] = {}
    
    def _get_default_translations(self, locale: str) -> Dict[str, str]:
        """Get default translations for a locale."""
        if locale == "en":
            return {
                # System messages
                "system.starting": "BCI-Agent-Bridge is starting...",
                "system.ready": "System ready",
                "system.stopping": "System stopping...",
                "system.stopped": "System stopped",
                "system.error": "System error occurred",
                
                # Device messages
                "device.connecting": "Connecting to BCI device...",
                "device.connected": "Device connected successfully",
                "device.disconnected": "Device disconnected",
                "device.error": "Device error",
                "device.calibrating": "Calibrating device...",
                "device.calibration_complete": "Calibration complete",
                
                # Neural processing
                "neural.processing": "Processing neural signals...",
                "neural.signal_quality.good": "Signal quality is good",
                "neural.signal_quality.poor": "Signal quality is poor",
                "neural.confidence.high": "High confidence detection",
                "neural.confidence.low": "Low confidence detection",
                "neural.paradigm.p300": "P300 paradigm active",
                "neural.paradigm.motor_imagery": "Motor imagery paradigm active", 
                "neural.paradigm.ssvep": "SSVEP paradigm active",
                
                # Commands
                "command.move_left": "Move left",
                "command.move_right": "Move right",
                "command.move_forward": "Move forward",
                "command.move_backward": "Move backward",
                "command.select": "Select",
                "command.cancel": "Cancel",
                "command.yes": "Yes",
                "command.no": "No",
                
                # Medical/Safety
                "medical.emergency": "Emergency",
                "medical.pain": "Pain",
                "medical.help": "Help",
                "medical.stop": "Stop",
                "medical.break_needed": "Break needed",
                "medical.session_too_long": "Session duration too long",
                "medical.fatigue_detected": "Fatigue detected",
                
                # Alerts
                "alert.low_signal": "Low signal quality detected",
                "alert.high_latency": "High processing latency detected", 
                "alert.system_overload": "System overload detected",
                "alert.connection_lost": "Connection lost",
                
                # Claude Integration
                "claude.processing": "Processing through Claude AI...",
                "claude.response_received": "Response received from Claude",
                "claude.error": "Claude processing error",
                "claude.safety_check": "Safety check in progress",
                
                # UI Elements
                "ui.start": "Start",
                "ui.stop": "Stop",
                "ui.pause": "Pause",
                "ui.resume": "Resume",
                "ui.settings": "Settings",
                "ui.help": "Help",
                "ui.about": "About",
                "ui.language": "Language",
                "ui.close": "Close",
                "ui.save": "Save",
                "ui.cancel": "Cancel",
                
                # Status messages
                "status.healthy": "Healthy",
                "status.degraded": "Degraded",
                "status.unhealthy": "Unhealthy", 
                "status.offline": "Offline",
                "status.connecting": "Connecting",
                "status.connected": "Connected",
                "status.disconnected": "Disconnected",
                
                # Time/Duration
                "time.seconds": "seconds",
                "time.minutes": "minutes",
                "time.hours": "hours",
                "time.session_duration": "Session duration",
                "time.time_remaining": "Time remaining",
                
                # Metrics
                "metrics.accuracy": "Accuracy",
                "metrics.latency": "Latency",
                "metrics.throughput": "Throughput", 
                "metrics.signal_quality": "Signal quality",
                "metrics.confidence": "Confidence",
                "metrics.success_rate": "Success rate"
            }
        
        elif locale == "es":
            return {
                "system.starting": "BCI-Agent-Bridge está iniciando...",
                "system.ready": "Sistema listo",
                "system.stopping": "Sistema deteniéndose...",
                "system.stopped": "Sistema detenido",
                "device.connected": "Dispositivo conectado exitosamente",
                "neural.signal_quality.good": "La calidad de señal es buena",
                "neural.signal_quality.poor": "La calidad de señal es pobre",
                "command.move_left": "Mover izquierda",
                "command.move_right": "Mover derecha",
                "command.select": "Seleccionar",
                "command.yes": "Sí",
                "command.no": "No",
                "medical.emergency": "Emergencia",
                "medical.pain": "Dolor",
                "medical.help": "Ayuda",
                "medical.stop": "Parar",
                "ui.start": "Iniciar",
                "ui.stop": "Parar",
                "ui.settings": "Configuración",
                "ui.help": "Ayuda",
                "ui.language": "Idioma"
            }
        
        elif locale == "fr":
            return {
                "system.starting": "BCI-Agent-Bridge démarre...",
                "system.ready": "Système prêt",
                "system.stopping": "Arrêt du système...",
                "system.stopped": "Système arrêté",
                "device.connected": "Appareil connecté avec succès",
                "neural.signal_quality.good": "La qualité du signal est bonne",
                "neural.signal_quality.poor": "La qualité du signal est mauvaise",
                "command.move_left": "Aller à gauche",
                "command.move_right": "Aller à droite",
                "command.select": "Sélectionner",
                "command.yes": "Oui",
                "command.no": "Non",
                "medical.emergency": "Urgence",
                "medical.pain": "Douleur",
                "medical.help": "Aide",
                "medical.stop": "Arrêter",
                "ui.start": "Démarrer",
                "ui.stop": "Arrêter",
                "ui.settings": "Paramètres",
                "ui.help": "Aide",
                "ui.language": "Langue"
            }
        
        elif locale == "de":
            return {
                "system.starting": "BCI-Agent-Bridge startet...",
                "system.ready": "System bereit",
                "system.stopping": "System wird beendet...",
                "system.stopped": "System beendet",
                "device.connected": "Gerät erfolgreich verbunden",
                "neural.signal_quality.good": "Signalqualität ist gut",
                "neural.signal_quality.poor": "Signalqualität ist schlecht", 
                "command.move_left": "Nach links bewegen",
                "command.move_right": "Nach rechts bewegen",
                "command.select": "Auswählen",
                "command.yes": "Ja",
                "command.no": "Nein",
                "medical.emergency": "Notfall",
                "medical.pain": "Schmerz",
                "medical.help": "Hilfe",
                "medical.stop": "Stopp",
                "ui.start": "Starten",
                "ui.stop": "Stoppen",
                "ui.settings": "Einstellungen",
                "ui.help": "Hilfe",
                "ui.language": "Sprache"
            }
        
        elif locale == "ja":
            return {
                "system.starting": "BCI-Agent-Bridgeが開始しています...",
                "system.ready": "システム準備完了",
                "system.stopping": "システムを停止しています...",
                "system.stopped": "システム停止",
                "device.connected": "デバイス接続成功",
                "neural.signal_quality.good": "信号品質は良好です",
                "neural.signal_quality.poor": "信号品質が低下しています",
                "command.move_left": "左に移動",
                "command.move_right": "右に移動",
                "command.select": "選択",
                "command.yes": "はい",
                "command.no": "いいえ",
                "medical.emergency": "緊急事態",
                "medical.pain": "痛み",
                "medical.help": "助け",
                "medical.stop": "停止",
                "ui.start": "開始",
                "ui.stop": "停止",
                "ui.settings": "設定",
                "ui.help": "ヘルプ",
                "ui.language": "言語"
            }
        
        elif locale == "zh":
            return {
                "system.starting": "BCI-Agent-Bridge正在启动...",
                "system.ready": "系统就绪",
                "system.stopping": "系统正在停止...",
                "system.stopped": "系统已停止",
                "device.connected": "设备连接成功",
                "neural.signal_quality.good": "信号质量良好",
                "neural.signal_quality.poor": "信号质量较差",
                "command.move_left": "向左移动",
                "command.move_right": "向右移动",
                "command.select": "选择",
                "command.yes": "是",
                "command.no": "否",
                "medical.emergency": "紧急情况",
                "medical.pain": "疼痛",
                "medical.help": "帮助",
                "medical.stop": "停止",
                "ui.start": "开始",
                "ui.stop": "停止",
                "ui.settings": "设置",
                "ui.help": "帮助",
                "ui.language": "语言"
            }
        
        else:
            # Fallback to English for unsupported locales
            return self._get_default_translations("en")
    
    def set_locale(self, locale: str) -> bool:
        """Set current locale."""
        locale_info = get_locale_info(locale)
        
        with self._lock:
            if locale_info.code not in self.translations:
                self._load_locale_translations(locale_info.code)
            
            self.current_locale = locale_info.code
            self.logger.info(f"Locale set to {locale_info.code} ({locale_info.native_name})")
            return True
    
    def get_locale(self) -> str:
        """Get current locale."""
        return self.current_locale
    
    def translate(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Translate a key to the specified locale."""
        target_locale = locale or self.current_locale
        
        with self._lock:
            # Try target locale first
            if target_locale in self.translations:
                translation = self.translations[target_locale].get(key)
                if translation:
                    return self._format_translation(translation, **kwargs)
            
            # Fallback to English
            if DEFAULT_LOCALE in self.translations:
                translation = self.translations[DEFAULT_LOCALE].get(key)
                if translation:
                    return self._format_translation(translation, **kwargs)
            
            # Return key as fallback
            self.logger.warning(f"Translation not found for key '{key}' in locale '{target_locale}'")
            return key
    
    def _format_translation(self, translation: str, **kwargs) -> str:
        """Format translation with parameters."""
        try:
            return translation.format(**kwargs)
        except KeyError as e:
            self.logger.warning(f"Missing parameter {e} for translation: {translation}")
            return translation
        except Exception as e:
            self.logger.error(f"Error formatting translation: {e}")
            return translation
    
    def has_translation(self, key: str, locale: Optional[str] = None) -> bool:
        """Check if translation exists for key."""
        target_locale = locale or self.current_locale
        
        with self._lock:
            return (target_locale in self.translations and 
                   key in self.translations[target_locale])
    
    def add_translation(self, key: str, value: str, locale: Optional[str] = None) -> None:
        """Add or update a translation."""
        target_locale = locale or self.current_locale
        
        with self._lock:
            if target_locale not in self.translations:
                self.translations[target_locale] = {}
            
            self.translations[target_locale][key] = value
    
    def get_all_translations(self, locale: Optional[str] = None) -> Dict[str, str]:
        """Get all translations for a locale."""
        target_locale = locale or self.current_locale
        
        with self._lock:
            return self.translations.get(target_locale, {}).copy()
    
    def save_translations(self, locale: Optional[str] = None) -> bool:
        """Save translations to file."""
        target_locale = locale or self.current_locale
        
        if target_locale not in self.translations:
            return False
        
        translation_file = self.translations_dir / f"{target_locale}.json"
        
        try:
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(
                    self.translations[target_locale], 
                    f, 
                    indent=2, 
                    ensure_ascii=False,
                    sort_keys=True
                )
            
            self.logger.info(f"Saved translations for {target_locale}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save translations for {target_locale}: {e}")
            return False
    
    def reload_translations(self) -> None:
        """Reload all translations from files."""
        self._load_translations()
        self.logger.info("Reloaded all translations")
    
    def get_supported_locales(self) -> List[Dict[str, str]]:
        """Get list of supported locales with their info."""
        return [
            {
                "code": info.code,
                "name": info.name,
                "native_name": info.native_name,
                "rtl": info.rtl
            }
            for info in SUPPORTED_LOCALES.values()
        ]


# Global translation manager instance
_global_translator: Optional[TranslationManager] = None
_translator_lock = threading.Lock()


def get_translator() -> TranslationManager:
    """Get the global translator instance."""
    global _global_translator
    
    with _translator_lock:
        if _global_translator is None:
            # Try to get locale from environment
            import os
            locale = os.getenv("BCI_LOCALE", DEFAULT_LOCALE)
            _global_translator = TranslationManager(locale)
        
        return _global_translator


def set_global_locale(locale: str) -> bool:
    """Set the global locale."""
    translator = get_translator()
    return translator.set_locale(locale)


def _(key: str, **kwargs) -> str:
    """Shorthand function for translation."""
    translator = get_translator()
    return translator.translate(key, **kwargs)


# Context manager for temporary locale changes
class TemporaryLocale:
    """Context manager for temporary locale changes."""
    
    def __init__(self, locale: str):
        self.locale = locale
        self.original_locale = None
        self.translator = get_translator()
    
    def __enter__(self):
        self.original_locale = self.translator.get_locale()
        self.translator.set_locale(self.locale)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_locale:
            self.translator.set_locale(self.original_locale)