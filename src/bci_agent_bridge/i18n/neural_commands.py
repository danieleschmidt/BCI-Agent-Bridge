"""
Neural command translation and localization.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import re

from .translator import TranslationManager, get_translator
from .locales import get_medical_term, SUPPORTED_LOCALES


@dataclass
class CommandMapping:
    """Maps neural intention codes to localized commands."""
    code: str
    en_template: str
    priority: int = 0
    category: str = "general"
    medical: bool = False
    emergency: bool = False


class NeuralCommandTranslator:
    """Translates neural commands and intentions across languages."""
    
    def __init__(self, translator: Optional[TranslationManager] = None):
        self.translator = translator or get_translator()
        self.logger = logging.getLogger(__name__)
        
        # Command mappings for different paradigms
        self.command_mappings = self._initialize_command_mappings()
        
        # Cached translations for performance
        self._translation_cache: Dict[str, Dict[str, str]] = {}
    
    def _initialize_command_mappings(self) -> Dict[str, List[CommandMapping]]:
        """Initialize command mappings for different BCI paradigms."""
        return {
            "P300": [
                # Basic navigation
                CommandMapping("select", "command.select", 1, "navigation"),
                CommandMapping("cancel", "command.cancel", 1, "navigation"),
                CommandMapping("yes", "command.yes", 2, "decision"),
                CommandMapping("no", "command.no", 2, "decision"),
                CommandMapping("help", "medical.help", 3, "assistance", medical=True),
                CommandMapping("stop", "medical.stop", 3, "control", medical=True),
                
                # Communication
                CommandMapping("hello", "communication.hello", 0, "social"),
                CommandMapping("thank_you", "communication.thank_you", 0, "social"),
                CommandMapping("goodbye", "communication.goodbye", 0, "social"),
                
                # Medical/Emergency
                CommandMapping("emergency", "medical.emergency", 5, "emergency", medical=True, emergency=True),
                CommandMapping("pain", "medical.pain", 4, "medical", medical=True),
                CommandMapping("break", "medical.break_needed", 2, "comfort", medical=True),
                CommandMapping("tired", "medical.fatigue_detected", 2, "comfort", medical=True),
            ],
            
            "MotorImagery": [
                # Directional movement
                CommandMapping("move_left", "command.move_left", 1, "movement"),
                CommandMapping("move_right", "command.move_right", 1, "movement"),
                CommandMapping("move_forward", "command.move_forward", 1, "movement"),
                CommandMapping("move_backward", "command.move_backward", 1, "movement"),
                CommandMapping("move_up", "command.move_up", 1, "movement"),
                CommandMapping("move_down", "command.move_down", 1, "movement"),
                
                # Control commands
                CommandMapping("start", "ui.start", 2, "control"),
                CommandMapping("stop", "ui.stop", 2, "control"), 
                CommandMapping("pause", "ui.pause", 2, "control"),
                CommandMapping("resume", "ui.resume", 2, "control"),
                
                # Emergency
                CommandMapping("emergency_stop", "medical.emergency", 5, "emergency", medical=True, emergency=True),
            ],
            
            "SSVEP": [
                # Menu selection
                CommandMapping("option_1", "ui.option_1", 1, "selection"),
                CommandMapping("option_2", "ui.option_2", 1, "selection"),
                CommandMapping("option_3", "ui.option_3", 1, "selection"),
                CommandMapping("option_4", "ui.option_4", 1, "selection"),
                
                # Frequency-based commands
                CommandMapping("freq_6hz", "ssvep.frequency_6", 0, "frequency"),
                CommandMapping("freq_7_5hz", "ssvep.frequency_7_5", 0, "frequency"),
                CommandMapping("freq_8_5hz", "ssvep.frequency_8_5", 0, "frequency"),
                CommandMapping("freq_10hz", "ssvep.frequency_10", 0, "frequency"),
                
                # Navigation
                CommandMapping("menu_up", "ui.menu_up", 1, "navigation"),
                CommandMapping("menu_down", "ui.menu_down", 1, "navigation"),
                CommandMapping("menu_select", "ui.select", 1, "navigation"),
                CommandMapping("menu_back", "ui.back", 1, "navigation"),
            ]
        }
    
    def translate_command(self, command_code: str, paradigm: str, 
                         locale: Optional[str] = None, **kwargs) -> str:
        """Translate a neural command code to localized text."""
        target_locale = locale or self.translator.get_locale()
        
        # Check cache first
        cache_key = f"{command_code}_{paradigm}_{target_locale}"
        if cache_key in self._translation_cache:
            cached_translation = self._translation_cache[cache_key]
            if cached_translation:
                return self._format_command(cached_translation, **kwargs)
        
        # Find command mapping
        command_mapping = self._find_command_mapping(command_code, paradigm)
        if not command_mapping:
            self.logger.warning(f"No mapping found for command '{command_code}' in paradigm '{paradigm}'")
            return command_code
        
        # Translate using the mapping
        translation = self.translator.translate(
            command_mapping.en_template, 
            locale=target_locale
        )
        
        # Cache the translation
        if cache_key not in self._translation_cache:
            self._translation_cache[cache_key] = translation
        
        return self._format_command(translation, **kwargs)
    
    def _find_command_mapping(self, command_code: str, paradigm: str) -> Optional[CommandMapping]:
        """Find command mapping for given code and paradigm."""
        if paradigm not in self.command_mappings:
            return None
        
        for mapping in self.command_mappings[paradigm]:
            if mapping.code == command_code:
                return mapping
        
        return None
    
    def _format_command(self, translation: str, **kwargs) -> str:
        """Format command translation with parameters."""
        try:
            return translation.format(**kwargs)
        except Exception as e:
            self.logger.warning(f"Error formatting command translation: {e}")
            return translation
    
    def get_available_commands(self, paradigm: str, 
                             category: Optional[str] = None,
                             include_emergency: bool = True) -> List[Dict[str, any]]:
        """Get available commands for a paradigm."""
        if paradigm not in self.command_mappings:
            return []
        
        commands = []
        current_locale = self.translator.get_locale()
        
        for mapping in self.command_mappings[paradigm]:
            # Filter by category if specified
            if category and mapping.category != category:
                continue
            
            # Filter emergency commands if not requested
            if mapping.emergency and not include_emergency:
                continue
            
            translation = self.translator.translate(mapping.en_template)
            
            commands.append({
                "code": mapping.code,
                "text": translation,
                "category": mapping.category,
                "priority": mapping.priority,
                "medical": mapping.medical,
                "emergency": mapping.emergency,
                "locale": current_locale
            })
        
        # Sort by priority (higher first) then alphabetically
        commands.sort(key=lambda x: (-x["priority"], x["text"]))
        return commands
    
    def detect_language_from_text(self, text: str) -> Optional[str]:
        """Attempt to detect language from neural command text."""
        # Simple language detection based on medical terms
        text_lower = text.lower()
        
        # Score each language based on matching terms
        language_scores = {}
        
        for locale_code in SUPPORTED_LOCALES.keys():
            score = 0
            
            # Check for medical terms in this language
            for en_term in ["emergency", "pain", "help", "stop", "yes", "no"]:
                localized_term = get_medical_term(en_term, locale_code)
                if localized_term.lower() in text_lower:
                    score += 2
                
                # Partial matching
                if len(localized_term) > 3 and localized_term.lower()[:3] in text_lower:
                    score += 1
            
            # Check for common command words
            common_commands = self.get_available_commands("P300")
            for cmd in common_commands:
                cmd_text = self.translator.translate(cmd["code"], locale=locale_code)
                if cmd_text.lower() in text_lower:
                    score += 1
            
            if score > 0:
                language_scores[locale_code] = score
        
        if not language_scores:
            return None
        
        # Return language with highest score
        best_language = max(language_scores, key=language_scores.get)
        confidence = language_scores[best_language]
        
        # Require minimum confidence
        if confidence >= 2:
            return best_language
        
        return None
    
    def translate_neural_intention(self, intention_text: str, 
                                 source_locale: Optional[str] = None,
                                 target_locale: Optional[str] = None) -> Dict[str, any]:
        """Translate neural intention from one language to another."""
        target_locale = target_locale or self.translator.get_locale()
        
        # Auto-detect source language if not provided
        if not source_locale:
            detected_locale = self.detect_language_from_text(intention_text)
            source_locale = detected_locale or "en"
        
        # If source and target are the same, return as-is
        if source_locale == target_locale:
            return {
                "original_text": intention_text,
                "translated_text": intention_text,
                "source_locale": source_locale,
                "target_locale": target_locale,
                "confidence": 1.0,
                "medical_terms": self._extract_medical_terms(intention_text, source_locale)
            }
        
        # Extract and translate medical terms
        medical_terms = self._extract_medical_terms(intention_text, source_locale)
        translated_terms = {}
        
        for term, positions in medical_terms.items():
            translated_term = get_medical_term(term, target_locale)
            translated_terms[translated_term] = positions
        
        # Perform basic translation by replacing medical terms
        translated_text = intention_text
        for original_term, translated_term in zip(medical_terms.keys(), translated_terms.keys()):
            translated_text = re.sub(
                r'\b' + re.escape(original_term) + r'\b',
                translated_term,
                translated_text,
                flags=re.IGNORECASE
            )
        
        return {
            "original_text": intention_text,
            "translated_text": translated_text,
            "source_locale": source_locale,
            "target_locale": target_locale,
            "confidence": 0.8 if medical_terms else 0.3,
            "medical_terms": translated_terms
        }
    
    def _extract_medical_terms(self, text: str, locale: str) -> Dict[str, List[int]]:
        """Extract medical terms and their positions from text."""
        medical_terms = {}
        text_lower = text.lower()
        
        # Check for known medical terms in the given locale
        for en_term in ["emergency", "pain", "help", "stop", "yes", "no", 
                       "doctor", "nurse", "medication", "therapy"]:
            localized_term = get_medical_term(en_term, locale)
            
            # Find all occurrences
            positions = []
            start = 0
            while True:
                pos = text_lower.find(localized_term.lower(), start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            
            if positions:
                medical_terms[localized_term] = positions
        
        return medical_terms
    
    def create_localized_command_menu(self, paradigm: str, 
                                    locale: Optional[str] = None) -> Dict[str, any]:
        """Create a localized command menu structure."""
        target_locale = locale or self.translator.get_locale()
        locale_info = self.translator.translate("ui.language", locale=target_locale)
        
        commands = self.get_available_commands(paradigm, include_emergency=True)
        
        # Group commands by category
        categories = {}
        for cmd in commands:
            category = cmd["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(cmd)
        
        return {
            "paradigm": paradigm,
            "locale": target_locale,
            "locale_info": locale_info,
            "total_commands": len(commands),
            "categories": categories,
            "emergency_commands": [cmd for cmd in commands if cmd["emergency"]],
            "medical_commands": [cmd for cmd in commands if cmd["medical"]],
            "created_at": self.translator.translate("time.created_at", locale=target_locale)
        }
    
    def validate_neural_command(self, command_text: str, paradigm: str,
                              expected_locale: Optional[str] = None) -> Dict[str, any]:
        """Validate and analyze a neural command."""
        expected_locale = expected_locale or self.translator.get_locale()
        
        # Check if command exists in current paradigm
        available_commands = self.get_available_commands(paradigm)
        command_found = False
        matching_command = None
        
        for cmd in available_commands:
            if cmd["text"].lower() == command_text.lower():
                command_found = True
                matching_command = cmd
                break
        
        # Detect actual language
        detected_locale = self.detect_language_from_text(command_text)
        
        # Extract medical terms
        medical_terms = self._extract_medical_terms(command_text, expected_locale)
        
        return {
            "valid": command_found,
            "command": matching_command,
            "expected_locale": expected_locale,
            "detected_locale": detected_locale,
            "locale_match": detected_locale == expected_locale if detected_locale else None,
            "medical_terms": medical_terms,
            "is_emergency": matching_command and matching_command.get("emergency", False) if matching_command else False,
            "is_medical": matching_command and matching_command.get("medical", False) if matching_command else False,
            "suggestions": self._get_command_suggestions(command_text, paradigm, expected_locale)
        }
    
    def _get_command_suggestions(self, command_text: str, paradigm: str, 
                               locale: str) -> List[str]:
        """Get command suggestions for similar text."""
        available_commands = self.get_available_commands(paradigm)
        suggestions = []
        
        command_lower = command_text.lower()
        
        # Find commands that start with the same letters
        for cmd in available_commands:
            cmd_text_lower = cmd["text"].lower()
            
            # Exact substring match
            if command_lower in cmd_text_lower or cmd_text_lower in command_lower:
                suggestions.append(cmd["text"])
            
            # First word match
            elif command_lower.split()[0] in cmd_text_lower.split() if command_lower.split() else False:
                suggestions.append(cmd["text"])
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def clear_translation_cache(self) -> None:
        """Clear the translation cache."""
        self._translation_cache.clear()
        self.logger.info("Neural command translation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get translation cache statistics."""
        return {
            "cache_size": len(self._translation_cache),
            "cached_locales": list(set(key.split("_")[-1] for key in self._translation_cache.keys())),
            "supported_paradigms": list(self.command_mappings.keys()),
            "total_command_mappings": sum(len(mappings) for mappings in self.command_mappings.values())
        }