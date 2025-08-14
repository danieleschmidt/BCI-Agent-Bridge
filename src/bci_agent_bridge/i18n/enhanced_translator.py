"""
Enhanced Global Translation and Localization System for BCI-Agent-Bridge.
Supports real-time neural command translation across multiple languages and cultures.
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import uuid
from datetime import datetime
import re

# Core imports
try:
    from .locales import SUPPORTED_LOCALES, LANGUAGE_CODES, REGION_CODES
    from .neural_commands import NEURAL_COMMAND_TRANSLATIONS
    _LOCALES_AVAILABLE = True
except ImportError:
    _LOCALES_AVAILABLE = False
    SUPPORTED_LOCALES = {}
    LANGUAGE_CODES = {}
    REGION_CODES = {}
    NEURAL_COMMAND_TRANSLATIONS = {}

logger = logging.getLogger(__name__)


class TranslationMode(Enum):
    """Translation modes for different use cases."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    ADAPTIVE = "adaptive"
    CLINICAL = "clinical"
    EMERGENCY = "emergency"


class LocalizationLevel(Enum):
    """Levels of localization support."""
    BASIC = "basic"          # Language only
    REGIONAL = "regional"    # Language + region
    CULTURAL = "cultural"    # Language + region + cultural context
    MEDICAL = "medical"      # Medical terminology adaptation
    ACCESSIBLE = "accessible" # Accessibility features


@dataclass
class TranslationContext:
    """Context for neural command translations."""
    user_id: str
    session_id: str
    language: str
    region: Optional[str] = None
    medical_context: Optional[str] = None
    urgency_level: str = "normal"  # normal, urgent, emergency
    accessibility_needs: List[str] = field(default_factory=list)
    cultural_preferences: Dict[str, Any] = field(default_factory=dict)
    technical_level: str = "standard"  # basic, standard, technical


@dataclass
class TranslationResult:
    """Result of translation operation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    context: TranslationContext
    translation_time_ms: float
    alternatives: List[str] = field(default_factory=list)
    cultural_adaptations: Dict[str, str] = field(default_factory=list)
    accessibility_versions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NeuralCommandTranslator:
    """
    Advanced neural command translator with cultural awareness,
    medical terminology, and real-time adaptation.
    """
    
    def __init__(
        self,
        default_language: str = "en",
        translation_mode: TranslationMode = TranslationMode.ADAPTIVE,
        localization_level: LocalizationLevel = LocalizationLevel.CULTURAL
    ):
        self.default_language = default_language
        self.translation_mode = translation_mode
        self.localization_level = localization_level
        
        # Translation databases
        self.neural_commands = self._load_neural_commands()
        self.medical_terminology = self._load_medical_terminology()
        self.cultural_adaptations = self._load_cultural_adaptations()
        self.emergency_phrases = self._load_emergency_phrases()
        
        # Language models and caches
        self.translation_cache = {}
        self.context_aware_translations = defaultdict(dict)
        self.user_preferences = {}
        
        # Performance tracking
        self.translation_stats = {
            'total_translations': 0,
            'cache_hits': 0,
            'avg_translation_time_ms': 0.0,
            'languages_supported': len(SUPPORTED_LOCALES),
            'accuracy_scores': defaultdict(list)
        }
        
        # Real-time adaptation
        self.adaptive_learning = True
        self.user_feedback_weights = defaultdict(float)
        self.context_patterns = defaultdict(list)
        
        logger.info(f"Enhanced translator initialized: {translation_mode.value} mode, {localization_level.value} level")
    
    def _load_neural_commands(self) -> Dict[str, Dict[str, str]]:
        """Load neural command translations."""
        if _LOCALES_AVAILABLE:
            return NEURAL_COMMAND_TRANSLATIONS.copy()
        
        # Fallback neural commands database
        return {
            'select_item': {
                'en': 'Select current item',
                'es': 'Seleccionar elemento actual',
                'fr': 'Sélectionner l\'élément actuel',
                'de': 'Aktuelles Element auswählen',
                'ja': '現在のアイテムを選択',
                'zh': '选择当前项目',
                'pt': 'Selecionar item atual',
                'it': 'Seleziona elemento corrente',
                'ru': 'Выбрать текущий элемент',
                'ar': 'حدد العنصر الحالي'
            },
            'move_left': {
                'en': 'Move left',
                'es': 'Mover a la izquierda',
                'fr': 'Déplacer vers la gauche',
                'de': 'Nach links bewegen',
                'ja': '左に移動',
                'zh': '向左移动',
                'pt': 'Mover para a esquerda',
                'it': 'Sposta a sinistra',
                'ru': 'Переместить влево',
                'ar': 'تحرك يسارا'
            },
            'move_right': {
                'en': 'Move right',
                'es': 'Mover a la derecha',
                'fr': 'Déplacer vers la droite',
                'de': 'Nach rechts bewegen',
                'ja': '右に移動',
                'zh': '向右移动',
                'pt': 'Mover para a direita',
                'it': 'Sposta a destra',
                'ru': 'Переместить вправо',
                'ar': 'تحرك يمينا'
            },
            'confirm_action': {
                'en': 'Confirm action',
                'es': 'Confirmar acción',
                'fr': 'Confirmer l\'action',
                'de': 'Aktion bestätigen',
                'ja': 'アクションを確認',
                'zh': '确认操作',
                'pt': 'Confirmar ação',
                'it': 'Conferma azione',
                'ru': 'Подтвердить действие',
                'ar': 'تأكيد الإجراء'
            },
            'cancel_operation': {
                'en': 'Cancel operation',
                'es': 'Cancelar operación',
                'fr': 'Annuler l\'opération',
                'de': 'Vorgang abbrechen',
                'ja': '操作をキャンセル',
                'zh': '取消操作',
                'pt': 'Cancelar operação',
                'it': 'Annulla operazione',
                'ru': 'Отменить операцию',
                'ar': 'إلغاء العملية'
            }
        }
    
    def _load_medical_terminology(self) -> Dict[str, Dict[str, str]]:
        """Load medical terminology translations."""
        return {
            'emergency_stop': {
                'en': 'Emergency stop',
                'es': 'Parada de emergencia',
                'fr': 'Arrêt d\'urgence',
                'de': 'Notaus',
                'ja': '緊急停止',
                'zh': '紧急停止',
                'pt': 'Parada de emergência',
                'it': 'Arresto di emergenza',
                'ru': 'Аварийная остановка',
                'ar': 'توقف طارئ'
            },
            'pain_level': {
                'en': 'Pain level',
                'es': 'Nivel de dolor',
                'fr': 'Niveau de douleur',
                'de': 'Schmerzlevel',
                'ja': '痛みレベル',
                'zh': '疼痛程度',
                'pt': 'Nível de dor',
                'it': 'Livello di dolore',
                'ru': 'Уровень боли',
                'ar': 'مستوى الألم'
            },
            'medication_request': {
                'en': 'Request medication',
                'es': 'Solicitar medicación',
                'fr': 'Demander des médicaments',
                'de': 'Medikament anfordern',
                'ja': '薬を要求',
                'zh': '请求药物',
                'pt': 'Solicitar medicação',
                'it': 'Richiedere farmaci',
                'ru': 'Запросить лекарство',
                'ar': 'طلب الدواء'
            },
            'call_nurse': {
                'en': 'Call nurse',
                'es': 'Llamar enfermera',
                'fr': 'Appeler l\'infirmière',
                'de': 'Krankenschwester rufen',
                'ja': '看護師を呼ぶ',
                'zh': '呼叫护士',
                'pt': 'Chamar enfermeira',
                'it': 'Chiamare l\'infermiera',
                'ru': 'Вызвать медсестру',
                'ar': 'استدعاء الممرضة'
            }
        }
    
    def _load_cultural_adaptations(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Load cultural adaptations for different regions."""
        return {
            'greetings': {
                'formal': {
                    'en': 'Good morning',
                    'es': 'Buenos días',
                    'fr': 'Bonjour',
                    'de': 'Guten Morgen',
                    'ja': 'おはようございます',
                    'zh': '早上好',
                    'pt': 'Bom dia',
                    'it': 'Buongiorno',
                    'ru': 'Доброе утро',
                    'ar': 'صباح الخير'
                },
                'casual': {
                    'en': 'Hi there',
                    'es': 'Hola',
                    'fr': 'Salut',
                    'de': 'Hallo',
                    'ja': 'こんにちは',
                    'zh': '你好',
                    'pt': 'Olá',
                    'it': 'Ciao',
                    'ru': 'Привет',
                    'ar': 'مرحبا'
                }
            },
            'politeness_markers': {
                'please': {
                    'en': 'please',
                    'es': 'por favor',
                    'fr': 's\'il vous plaît',
                    'de': 'bitte',
                    'ja': 'お願いします',
                    'zh': '请',
                    'pt': 'por favor',
                    'it': 'per favore',
                    'ru': 'пожалуйста',
                    'ar': 'من فضلك'
                },
                'thank_you': {
                    'en': 'thank you',
                    'es': 'gracias',
                    'fr': 'merci',
                    'de': 'danke',
                    'ja': 'ありがとう',
                    'zh': '谢谢',
                    'pt': 'obrigado',
                    'it': 'grazie',
                    'ru': 'спасибо',
                    'ar': 'شكرا'
                }
            }
        }
    
    def _load_emergency_phrases(self) -> Dict[str, Dict[str, str]]:
        """Load emergency phrases for critical situations."""
        return {
            'help': {
                'en': 'HELP!',
                'es': '¡AYUDA!',
                'fr': 'À L\'AIDE!',
                'de': 'HILFE!',
                'ja': '助けて！',
                'zh': '救命！',
                'pt': 'SOCORRO!',
                'it': 'AIUTO!',
                'ru': 'ПОМОЩЬ!',
                'ar': '!مساعدة'
            },
            'pain': {
                'en': 'I\'m in pain',
                'es': 'Tengo dolor',
                'fr': 'J\'ai mal',
                'de': 'Ich habe Schmerzen',
                'ja': '痛いです',
                'zh': '我很痛',
                'pt': 'Estou com dor',
                'it': 'Ho dolore',
                'ru': 'Мне больно',
                'ar': 'أشعر بألم'
            },
            'emergency': {
                'en': 'This is an emergency',
                'es': 'Esto es una emergencia',
                'fr': 'C\'est une urgence',
                'de': 'Das ist ein Notfall',
                'ja': 'これは緊急事態です',
                'zh': '这是紧急情况',
                'pt': 'Esta é uma emergência',
                'it': 'Questa è un\'emergenza',
                'ru': 'Это экстренная ситуация',
                'ar': 'هذه حالة طوارئ'
            }
        }
    
    async def translate_neural_command(
        self,
        command: str,
        context: TranslationContext
    ) -> TranslationResult:
        """
        Translate neural command with full cultural and medical context.
        
        Args:
            command: Neural command to translate
            context: Translation context including language, region, medical needs
            
        Returns:
            Comprehensive translation result
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(command, context)
            if cache_key in self.translation_cache:
                cached_result = self.translation_cache[cache_key]
                self.translation_stats['cache_hits'] += 1
                return cached_result
            
            # Determine translation strategy
            if context.urgency_level == "emergency":
                translation_strategy = "emergency"
            elif context.medical_context:
                translation_strategy = "medical"
            elif self.translation_mode == TranslationMode.CLINICAL:
                translation_strategy = "clinical"
            else:
                translation_strategy = "standard"
            
            # Perform base translation
            base_translation = await self._translate_base_command(command, context)
            
            # Apply cultural adaptations
            cultural_translation = await self._apply_cultural_adaptations(
                base_translation, context, translation_strategy
            )
            
            # Generate accessibility versions
            accessibility_versions = await self._generate_accessibility_versions(
                cultural_translation, context
            )
            
            # Generate alternatives
            alternatives = await self._generate_translation_alternatives(
                command, context, cultural_translation
            )
            
            # Calculate confidence score
            confidence = self._calculate_translation_confidence(
                command, cultural_translation, context
            )
            
            # Create result
            translation_time = (time.time() - start_time) * 1000
            result = TranslationResult(
                original_text=command,
                translated_text=cultural_translation,
                source_language=self._detect_source_language(command),
                target_language=context.language,
                confidence=confidence,
                context=context,
                translation_time_ms=translation_time,
                alternatives=alternatives,
                accessibility_versions=accessibility_versions,
                metadata={
                    'strategy': translation_strategy,
                    'cache_miss': True,
                    'adaptations_applied': True
                }
            )
            
            # Cache result
            self.translation_cache[cache_key] = result
            
            # Update statistics
            self._update_translation_stats(result)
            
            # Adaptive learning
            if self.adaptive_learning:
                await self._update_adaptive_models(command, result, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            
            # Return fallback translation
            return TranslationResult(
                original_text=command,
                translated_text=command,  # Fallback to original
                source_language="unknown",
                target_language=context.language,
                confidence=0.0,
                context=context,
                translation_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e), 'fallback': True}
            )
    
    async def _translate_base_command(self, command: str, context: TranslationContext) -> str:
        """Perform base translation of neural command."""
        # Normalize command
        normalized_command = self._normalize_command(command)
        
        # Look up in neural commands database
        if normalized_command in self.neural_commands:
            translations = self.neural_commands[normalized_command]
            if context.language in translations:
                return translations[context.language]
        
        # Look up in medical terminology if medical context
        if context.medical_context and normalized_command in self.medical_terminology:
            translations = self.medical_terminology[normalized_command]
            if context.language in translations:
                return translations[context.language]
        
        # Look up in emergency phrases if urgent
        if context.urgency_level == "emergency" and normalized_command in self.emergency_phrases:
            translations = self.emergency_phrases[normalized_command]
            if context.language in translations:
                return translations[context.language]
        
        # Fallback to pattern-based translation
        return await self._pattern_based_translation(command, context)
    
    def _normalize_command(self, command: str) -> str:
        """Normalize command for database lookup."""
        # Convert to lowercase
        normalized = command.lower().strip()
        
        # Remove common prefixes/suffixes
        prefixes = ['please ', 'i want to ', 'can you ', 'let me ']
        suffixes = [' please', ' now', ' immediately']
        
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break
        
        # Map common variations
        command_mappings = {
            'select': 'select_item',
            'choose': 'select_item',
            'pick': 'select_item',
            'left': 'move_left',
            'go left': 'move_left',
            'right': 'move_right',
            'go right': 'move_right',
            'yes': 'confirm_action',
            'ok': 'confirm_action',
            'confirm': 'confirm_action',
            'no': 'cancel_operation',
            'cancel': 'cancel_operation',
            'stop': 'cancel_operation'
        }
        
        return command_mappings.get(normalized, normalized)
    
    async def _pattern_based_translation(self, command: str, context: TranslationContext) -> str:
        """Fallback pattern-based translation."""
        # Simple word-by-word translation patterns
        word_translations = {
            'en_to_es': {
                'select': 'seleccionar',
                'move': 'mover',
                'left': 'izquierda',
                'right': 'derecha',
                'up': 'arriba',
                'down': 'abajo',
                'confirm': 'confirmar',
                'cancel': 'cancelar',
                'help': 'ayuda',
                'stop': 'parar'
            },
            'en_to_fr': {
                'select': 'sélectionner',
                'move': 'déplacer',
                'left': 'gauche',
                'right': 'droite',
                'up': 'haut',
                'down': 'bas',
                'confirm': 'confirmer',
                'cancel': 'annuler',
                'help': 'aide',
                'stop': 'arrêter'
            }
        }
        
        # Try pattern-based translation
        pattern_key = f"en_to_{context.language}"
        if pattern_key in word_translations:
            words = command.lower().split()
            translated_words = []
            
            for word in words:
                translated_word = word_translations[pattern_key].get(word, word)
                translated_words.append(translated_word)
            
            return ' '.join(translated_words)
        
        # Ultimate fallback - return original command
        return command
    
    async def _apply_cultural_adaptations(
        self,
        base_translation: str,
        context: TranslationContext,
        strategy: str
    ) -> str:
        """Apply cultural adaptations to base translation."""
        if self.localization_level in [LocalizationLevel.BASIC, LocalizationLevel.REGIONAL]:
            return base_translation
        
        adapted_translation = base_translation
        
        # Apply politeness markers based on cultural context
        cultural_prefs = context.cultural_preferences
        
        # Add politeness for formal cultures
        formal_cultures = ['ja', 'de', 'fr']  # Japanese, German, French tend to be more formal
        if context.language in formal_cultures or cultural_prefs.get('formality', 'standard') == 'high':
            adapted_translation = self._add_politeness_markers(adapted_translation, context.language)
        
        # Medical context adaptations
        if strategy in ['medical', 'clinical', 'emergency']:
            adapted_translation = self._apply_medical_adaptations(adapted_translation, context)
        
        # Regional variations
        if context.region:
            adapted_translation = self._apply_regional_variations(adapted_translation, context)
        
        return adapted_translation
    
    def _add_politeness_markers(self, translation: str, language: str) -> str:
        """Add appropriate politeness markers."""
        politeness = self.cultural_adaptations.get('politeness_markers', {})
        
        if 'please' in politeness and language in politeness['please']:
            please_marker = politeness['please'][language]
            
            # Add "please" for imperative commands
            if any(verb in translation.lower() for verb in ['select', 'move', 'confirm', 'cancel']):
                return f"{please_marker} {translation}"
        
        return translation
    
    def _apply_medical_adaptations(self, translation: str, context: TranslationContext) -> str:
        """Apply medical context adaptations."""
        # Add urgency indicators for emergency situations
        if context.urgency_level == "emergency":
            urgency_markers = {
                'en': 'URGENT:',
                'es': 'URGENTE:',
                'fr': 'URGENT:',
                'de': 'DRINGEND:',
                'ja': '緊急:',
                'zh': '紧急:',
                'pt': 'URGENTE:',
                'it': 'URGENTE:',
                'ru': 'СРОЧНО:',
                'ar': ':عاجل'
            }
            
            marker = urgency_markers.get(context.language, 'URGENT:')
            return f"{marker} {translation}"
        
        # Add medical precision for clinical contexts
        if context.medical_context == "clinical_trial":
            return f"[CLINICAL] {translation}"
        
        return translation
    
    def _apply_regional_variations(self, translation: str, context: TranslationContext) -> str:
        """Apply regional language variations."""
        # Spanish regional variations
        if context.language == 'es':
            if context.region in ['mx', 'ar', 'cl']:  # Latin American Spanish
                # Convert some European Spanish terms to Latin American
                translation = translation.replace('ordenador', 'computadora')
                translation = translation.replace('móvil', 'celular')
        
        # English regional variations
        elif context.language == 'en':
            if context.region == 'gb':  # British English
                translation = translation.replace('color', 'colour')
                translation = translation.replace('center', 'centre')
        
        # French regional variations
        elif context.language == 'fr':
            if context.region == 'ca':  # Canadian French
                translation = translation.replace('e-mail', 'courriel')
        
        return translation
    
    async def _generate_accessibility_versions(
        self,
        translation: str,
        context: TranslationContext
    ) -> Dict[str, str]:
        """Generate accessibility-friendly versions."""
        accessibility_versions = {}
        
        for need in context.accessibility_needs:
            if need == "visual_impairment":
                # More descriptive version for screen readers
                accessibility_versions["screen_reader"] = self._enhance_for_screen_reader(translation)
            
            elif need == "hearing_impairment":
                # Visual emphasis version
                accessibility_versions["visual_emphasis"] = self._add_visual_emphasis(translation)
            
            elif need == "cognitive_impairment":
                # Simplified language version
                accessibility_versions["simplified"] = self._simplify_language(translation, context.language)
            
            elif need == "motor_impairment":
                # Abbreviated version for faster communication
                accessibility_versions["abbreviated"] = self._create_abbreviation(translation, context.language)
        
        return accessibility_versions
    
    def _enhance_for_screen_reader(self, text: str) -> str:
        """Enhance text for screen reader accessibility."""
        # Add descriptive elements
        enhanced = text
        
        # Spell out abbreviations
        abbreviations = {
            'OK': 'okay',
            'ID': 'identification',
            'AI': 'artificial intelligence'
        }
        
        for abbrev, full_form in abbreviations.items():
            enhanced = enhanced.replace(abbrev, full_form)
        
        return enhanced
    
    def _add_visual_emphasis(self, text: str) -> str:
        """Add visual emphasis for hearing impaired users."""
        # Add emphasis markers
        if any(urgent in text.upper() for urgent in ['URGENT', 'EMERGENCY', 'HELP']):
            return f"⚠️ {text.upper()} ⚠️"
        else:
            return f"▶ {text}"
    
    def _simplify_language(self, text: str, language: str) -> str:
        """Simplify language for cognitive accessibility."""
        # Use simpler vocabulary
        simplifications = {
            'en': {
                'select': 'pick',
                'confirm': 'yes',
                'cancel': 'no',
                'navigate': 'go',
                'terminate': 'stop'
            },
            'es': {
                'seleccionar': 'elegir',
                'confirmar': 'sí',
                'cancelar': 'no',
                'navegar': 'ir'
            }
        }
        
        if language in simplifications:
            simplified = text
            for complex_word, simple_word in simplifications[language].items():
                simplified = simplified.replace(complex_word, simple_word)
            return simplified
        
        return text
    
    def _create_abbreviation(self, text: str, language: str) -> str:
        """Create abbreviated version for motor accessibility."""
        # Common abbreviations by language
        abbreviations = {
            'en': {
                'select current item': 'sel',
                'move left': '←',
                'move right': '→',
                'confirm action': '✓',
                'cancel operation': '✗'
            },
            'es': {
                'seleccionar elemento actual': 'sel',
                'mover a la izquierda': '←',
                'mover a la derecha': '→',
                'confirmar acción': '✓',
                'cancelar operación': '✗'
            }
        }
        
        if language in abbreviations:
            for full_phrase, abbrev in abbreviations[language].items():
                if full_phrase.lower() in text.lower():
                    return abbrev
        
        # Fallback: first letters of words
        words = text.split()
        if len(words) > 1:
            return ''.join(word[0].upper() for word in words if word)
        
        return text[:3].upper()  # First 3 characters
    
    async def _generate_translation_alternatives(
        self,
        original_command: str,
        context: TranslationContext,
        primary_translation: str
    ) -> List[str]:
        """Generate alternative translations."""
        alternatives = []
        
        # Formal/informal variations
        if context.language in ['es', 'de', 'fr', 'pt', 'it']:
            formal_variation = self._generate_formal_variation(primary_translation, context.language)
            if formal_variation != primary_translation:
                alternatives.append(formal_variation)
        
        # Regional variations
        if context.region:
            regional_variation = self._apply_regional_variations(primary_translation, context)
            if regional_variation != primary_translation:
                alternatives.append(regional_variation)
        
        # Technical level variations
        if context.technical_level == "technical":
            technical_variation = self._generate_technical_variation(primary_translation, context.language)
            if technical_variation != primary_translation:
                alternatives.append(technical_variation)
        
        return alternatives[:3]  # Limit to 3 alternatives
    
    def _generate_formal_variation(self, text: str, language: str) -> str:
        """Generate more formal variation of the text."""
        formal_replacements = {
            'es': {
                'mueve': 'desplace',
                'elige': 'seleccione',
                'para': 'detenga'
            },
            'de': {
                'geh': 'gehen Sie',
                'wähl': 'wählen Sie',
                'stopp': 'halten Sie an'
            },
            'fr': {
                'va': 'allez',
                'choisis': 'choisissez',
                'arrête': 'arrêtez'
            }
        }
        
        if language in formal_replacements:
            formal_text = text
            for informal, formal in formal_replacements[language].items():
                formal_text = formal_text.replace(informal, formal)
            return formal_text
        
        return text
    
    def _generate_technical_variation(self, text: str, language: str) -> str:
        """Generate technical variation with precise terminology."""
        technical_terms = {
            'en': {
                'select': 'execute selection on',
                'move': 'navigate to',
                'confirm': 'validate operation',
                'cancel': 'abort procedure'
            },
            'es': {
                'seleccionar': 'ejecutar selección en',
                'mover': 'navegar hacia',
                'confirmar': 'validar operación'
            }
        }
        
        if language in technical_terms:
            technical_text = text
            for simple, technical in technical_terms[language].items():
                technical_text = technical_text.replace(simple, technical)
            return technical_text
        
        return text
    
    def _calculate_translation_confidence(
        self,
        original: str,
        translation: str,
        context: TranslationContext
    ) -> float:
        """Calculate confidence score for translation."""
        confidence_factors = []
        
        # Dictionary coverage
        normalized_command = self._normalize_command(original)
        if normalized_command in self.neural_commands:
            confidence_factors.append(0.9)  # High confidence for known commands
        elif normalized_command in self.medical_terminology:
            confidence_factors.append(0.85)  # High confidence for medical terms
        else:
            confidence_factors.append(0.6)  # Medium confidence for pattern-based
        
        # Language support quality
        language_quality_scores = {
            'en': 1.0,    # Primary language
            'es': 0.95,   # Well-supported
            'fr': 0.95,   # Well-supported
            'de': 0.90,   # Well-supported
            'ja': 0.85,   # Good support
            'zh': 0.85,   # Good support
            'pt': 0.90,   # Good support
            'it': 0.90,   # Good support
            'ru': 0.80,   # Moderate support
            'ar': 0.75    # Basic support
        }
        
        language_score = language_quality_scores.get(context.language, 0.5)
        confidence_factors.append(language_score)
        
        # Context appropriateness
        if context.medical_context and normalized_command in self.medical_terminology:
            confidence_factors.append(0.95)  # Medical context match
        elif context.urgency_level == "emergency" and normalized_command in self.emergency_phrases:
            confidence_factors.append(0.95)  # Emergency context match
        else:
            confidence_factors.append(0.8)   # General context
        
        # User feedback (if available)
        user_feedback_score = self.user_feedback_weights.get(
            f"{context.user_id}_{context.language}", 0.8
        )
        confidence_factors.append(user_feedback_score)
        
        # Calculate weighted average
        return sum(confidence_factors) / len(confidence_factors)
    
    def _detect_source_language(self, text: str) -> str:
        """Detect source language of input text."""
        # Simple language detection based on character patterns
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'ja'  # Japanese/Chinese characters
        elif re.search(r'[\u0600-\u06FF]', text):
            return 'ar'  # Arabic
        elif re.search(r'[\u0400-\u04FF]', text):
            return 'ru'  # Cyrillic
        else:
            return 'en'  # Default to English
    
    def _generate_cache_key(self, command: str, context: TranslationContext) -> str:
        """Generate cache key for translation."""
        key_components = [
            command.lower().strip(),
            context.language,
            context.region or "",
            context.medical_context or "",
            context.urgency_level,
            context.technical_level,
            str(sorted(context.accessibility_needs)),
            str(sorted(context.cultural_preferences.items()))
        ]
        
        return "|".join(key_components)
    
    def _update_translation_stats(self, result: TranslationResult) -> None:
        """Update translation performance statistics."""
        self.translation_stats['total_translations'] += 1
        
        # Update average translation time
        current_avg = self.translation_stats['avg_translation_time_ms']
        total_translations = self.translation_stats['total_translations']
        
        self.translation_stats['avg_translation_time_ms'] = (
            (current_avg * (total_translations - 1) + result.translation_time_ms) / total_translations
        )
        
        # Track accuracy by language
        self.translation_stats['accuracy_scores'][result.target_language].append(result.confidence)
    
    async def _update_adaptive_models(
        self,
        original_command: str,
        result: TranslationResult,
        context: TranslationContext
    ) -> None:
        """Update adaptive learning models based on translation results."""
        # Store context patterns for learning
        pattern_key = f"{context.language}_{context.technical_level}_{context.urgency_level}"
        self.context_patterns[pattern_key].append({
            'command': original_command,
            'translation': result.translated_text,
            'confidence': result.confidence,
            'timestamp': time.time()
        })
        
        # Limit pattern history size
        if len(self.context_patterns[pattern_key]) > 1000:
            self.context_patterns[pattern_key] = self.context_patterns[pattern_key][-500:]
    
    async def provide_translation_feedback(
        self,
        translation_id: str,
        feedback_score: float,
        context: TranslationContext,
        comments: Optional[str] = None
    ) -> None:
        """Accept user feedback to improve future translations."""
        # Update user feedback weights
        user_key = f"{context.user_id}_{context.language}"
        current_weight = self.user_feedback_weights[user_key]
        
        # Exponential moving average
        alpha = 0.1  # Learning rate
        self.user_feedback_weights[user_key] = (
            alpha * feedback_score + (1 - alpha) * current_weight
        )
        
        logger.info(f"Updated feedback for user {context.user_id} in {context.language}: {feedback_score}")
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with metadata."""
        languages = []
        
        # Get from neural commands database
        if self.neural_commands:
            sample_command = list(self.neural_commands.values())[0]
            supported_langs = list(sample_command.keys())
        else:
            supported_langs = ['en', 'es', 'fr', 'de', 'ja', 'zh', 'pt', 'it', 'ru', 'ar']
        
        language_names = {
            'en': 'English',
            'es': 'Español',
            'fr': 'Français',
            'de': 'Deutsch',
            'ja': '日本語',
            'zh': '中文',
            'pt': 'Português',
            'it': 'Italiano',
            'ru': 'Русский',
            'ar': 'العربية'
        }
        
        for lang_code in supported_langs:
            languages.append({
                'code': lang_code,
                'name': language_names.get(lang_code, lang_code.upper()),
                'quality': 'high' if lang_code in ['en', 'es', 'fr', 'de'] else 'good',
                'medical_support': lang_code in self.medical_terminology.get('emergency_stop', {}),
                'emergency_support': lang_code in self.emergency_phrases.get('help', {})
            })
        
        return languages
    
    def get_translation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive translation statistics."""
        stats = self.translation_stats.copy()
        
        # Calculate average accuracy by language
        language_accuracies = {}
        for lang, scores in stats['accuracy_scores'].items():
            if scores:
                language_accuracies[lang] = {
                    'average_confidence': sum(scores) / len(scores),
                    'total_translations': len(scores),
                    'min_confidence': min(scores),
                    'max_confidence': max(scores)
                }
        
        stats['language_accuracies'] = language_accuracies
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / max(stats['total_translations'], 1) * 100
        )
        
        return stats


# Factory function for easy instantiation
def create_enhanced_translator(config: Optional[Dict[str, Any]] = None) -> NeuralCommandTranslator:
    """Create and configure an enhanced neural command translator."""
    config = config or {}
    
    mode_map = {
        'real_time': TranslationMode.REAL_TIME,
        'batch': TranslationMode.BATCH,
        'adaptive': TranslationMode.ADAPTIVE,
        'clinical': TranslationMode.CLINICAL,
        'emergency': TranslationMode.EMERGENCY
    }
    
    level_map = {
        'basic': LocalizationLevel.BASIC,
        'regional': LocalizationLevel.REGIONAL,
        'cultural': LocalizationLevel.CULTURAL,
        'medical': LocalizationLevel.MEDICAL,
        'accessible': LocalizationLevel.ACCESSIBLE
    }
    
    translation_mode = mode_map.get(config.get('mode', 'adaptive'), TranslationMode.ADAPTIVE)
    localization_level = level_map.get(config.get('level', 'cultural'), LocalizationLevel.CULTURAL)
    
    return NeuralCommandTranslator(
        default_language=config.get('default_language', 'en'),
        translation_mode=translation_mode,
        localization_level=localization_level
    )