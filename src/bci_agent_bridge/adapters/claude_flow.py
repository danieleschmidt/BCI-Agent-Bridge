"""
Claude Flow Adapter for processing neural intentions through Claude AI.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

import anthropic
from anthropic import Anthropic

from ..core.bridge import DecodedIntention


class SafetyMode(Enum):
    MEDICAL = "medical"
    STANDARD = "standard"
    RESEARCH = "research"


@dataclass 
class ClaudeResponse:
    content: str
    reasoning: Optional[str]
    confidence: float
    safety_flags: List[str]
    processing_time_ms: float
    tokens_used: int


class ClaudeFlowAdapter:
    """
    Adapter for processing BCI intentions through Claude AI with medical safety.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        safety_mode: str = "medical",
        max_tokens: int = 1000,
        temperature: float = 0.3
    ):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.safety_mode = SafetyMode(safety_mode)
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.logger = logging.getLogger(__name__)
        self.conversation_history: List[Dict[str, str]] = []
        self.safety_filters = self._initialize_safety_filters()
        
        self.logger.info(f"Initialized Claude adapter with {safety_mode} safety mode")
    
    def _initialize_safety_filters(self) -> Dict[str, Any]:
        """Initialize safety filters based on mode."""
        if self.safety_mode == SafetyMode.MEDICAL:
            return {
                "medical_terms": ["emergency", "pain", "help", "distress"],
                "prohibited_actions": ["medication", "diagnosis", "treatment"],
                "urgency_detection": True,
                "clinical_context": True
            }
        elif self.safety_mode == SafetyMode.STANDARD:
            return {
                "harmful_content": True,
                "privacy_protection": True
            }
        else:  # RESEARCH
            return {
                "data_collection": True,
                "experiment_tracking": True
            }
    
    async def execute(
        self, 
        intention: DecodedIntention,
        context: Optional[Dict[str, Any]] = None
    ) -> ClaudeResponse:
        """
        Execute a neural intention through Claude AI.
        
        Args:
            intention: Decoded neural intention
            context: Additional context for processing
            
        Returns:
            ClaudeResponse: Processed response from Claude
        """
        start_time = time.time()
        
        # Apply safety filters
        safety_check = self._safety_check(intention, context)
        if not safety_check["safe"]:
            return self._create_safety_response(safety_check["reason"])
        
        # Build prompt with context
        prompt = self._build_prompt(intention, context)
        
        try:
            # Call Claude API
            response = await self._call_claude_api(prompt)
            
            # Process response
            processed_response = self._process_response(response, intention)
            processed_response.processing_time_ms = (time.time() - start_time) * 1000
            
            # Update conversation history
            self._update_history(intention.command, processed_response.content)
            
            return processed_response
            
        except Exception as e:
            self.logger.error(f"Claude API error: {e}")
            return self._create_error_response(str(e))
    
    def _safety_check(
        self, 
        intention: DecodedIntention, 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform safety checks on the intention."""
        safety_flags = []
        
        if self.safety_mode == SafetyMode.MEDICAL:
            # Check for medical emergencies
            if any(term in intention.command.lower() 
                   for term in self.safety_filters["medical_terms"]):
                safety_flags.append("medical_urgency")
            
            # Check confidence threshold for medical applications
            if intention.confidence < 0.8:
                safety_flags.append("low_confidence")
        
        # Check for prohibited medical actions
        if any(action in intention.command.lower() 
               for action in self.safety_filters.get("prohibited_actions", [])):
            return {"safe": False, "reason": "prohibited_medical_action"}
        
        return {"safe": True, "flags": safety_flags}
    
    def _build_prompt(
        self, 
        intention: DecodedIntention, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build the prompt for Claude based on intention and context."""
        base_prompt = f"""
You are a medical-grade BCI assistant processing neural intentions with high precision.

NEURAL INTENTION: {intention.command}
CONFIDENCE: {intention.confidence:.3f}
PARADIGM: {intention.context.get('paradigm', 'Unknown')}
TIMESTAMP: {intention.timestamp}

SAFETY MODE: {self.safety_mode.value}
"""
        
        if context:
            base_prompt += f"\nADDITIONAL CONTEXT: {context}"
        
        if self.safety_mode == SafetyMode.MEDICAL:
            base_prompt += """
MEDICAL GUIDELINES:
- Prioritize patient safety and comfort
- Never provide medical diagnosis or treatment advice
- Escalate urgent situations immediately
- Maintain HIPAA compliance
- Use clear, reassuring communication
"""
        
        base_prompt += f"""
TASK: Process this neural intention and provide:
1. Appropriate response or action
2. Safety assessment
3. Confidence in interpretation
4. Next steps if applicable

Respond in JSON format:
{{
    "response": "your response here",
    "action": "recommended action",
    "safety_level": "safe|caution|urgent",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
"""
        
        return base_prompt
    
    async def _call_claude_api(self, prompt: str) -> anthropic.types.Message:
        """Make API call to Claude."""
        try:
            message = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return message
        except Exception as e:
            self.logger.error(f"Claude API call failed: {e}")
            raise
    
    def _process_response(
        self, 
        response: anthropic.types.Message, 
        intention: DecodedIntention
    ) -> ClaudeResponse:
        """Process the raw Claude response."""
        try:
            content = response.content[0].text
            
            # Try to parse JSON response
            import json
            try:
                parsed = json.loads(content)
                return ClaudeResponse(
                    content=parsed.get("response", content),
                    reasoning=parsed.get("reasoning"),
                    confidence=float(parsed.get("confidence", 0.5)),
                    safety_flags=self._extract_safety_flags(parsed),
                    processing_time_ms=0.0,  # Will be set by caller
                    tokens_used=response.usage.input_tokens + response.usage.output_tokens
                )
            except json.JSONDecodeError:
                # Fallback to raw content
                return ClaudeResponse(
                    content=content,
                    reasoning=None,
                    confidence=intention.confidence,
                    safety_flags=[],
                    processing_time_ms=0.0,
                    tokens_used=response.usage.input_tokens + response.usage.output_tokens
                )
                
        except Exception as e:
            self.logger.error(f"Response processing error: {e}")
            return self._create_error_response("Response processing failed")
    
    def _extract_safety_flags(self, parsed_response: Dict[str, Any]) -> List[str]:
        """Extract safety flags from parsed response."""
        flags = []
        safety_level = parsed_response.get("safety_level", "safe")
        
        if safety_level == "urgent":
            flags.append("urgent_attention")
        elif safety_level == "caution":
            flags.append("requires_caution")
        
        return flags
    
    def _create_safety_response(self, reason: str) -> ClaudeResponse:
        """Create a safety-blocked response."""
        return ClaudeResponse(
            content=f"Safety protocol activated: {reason}",
            reasoning="Blocked by safety filters",
            confidence=1.0,
            safety_flags=["safety_block"],
            processing_time_ms=0.0,
            tokens_used=0
        )
    
    def _create_error_response(self, error: str) -> ClaudeResponse:
        """Create an error response."""
        return ClaudeResponse(
            content=f"Processing error: {error}",
            reasoning="System error occurred",
            confidence=0.0,
            safety_flags=["system_error"],
            processing_time_ms=0.0,
            tokens_used=0
        )
    
    def _update_history(self, command: str, response: str) -> None:
        """Update conversation history."""
        self.conversation_history.append({
            "timestamp": time.time(),
            "command": command,
            "response": response
        })
        
        # Keep only last 50 interactions
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def process_text(self, text: str) -> ClaudeResponse:
        """Process plain text through Claude (for P300 speller)."""
        from ..core.bridge import DecodedIntention
        
        # Create a text-based intention
        intention = DecodedIntention(
            command=f"Process text: {text}",
            confidence=1.0,
            context={"type": "text_input"},
            timestamp=time.time()
        )
        
        return asyncio.run(self.execute(intention))
    
    def set_mode(self, mode: str) -> None:
        """Set processing mode (simplified, detailed, etc.)."""
        self.processing_mode = mode
        self.logger.info(f"Processing mode set to: {mode}")
    
    def suggest_break(self) -> ClaudeResponse:
        """Suggest a break to the user."""
        return ClaudeResponse(
            content="I notice you might be experiencing fatigue. Would you like to take a break?",
            reasoning="Fatigue detection triggered",
            confidence=0.9,
            safety_flags=["fatigue_detected"],
            processing_time_ms=0.0,
            tokens_used=0
        )
    
    def increase_engagement(self) -> ClaudeResponse:
        """Increase engagement when attention is low."""
        return ClaudeResponse(
            content="Let me help you refocus. What would you like to work on?",
            reasoning="Low attention detected",
            confidence=0.8,
            safety_flags=["attention_support"],
            processing_time_ms=0.0,
            tokens_used=0
        )
    
    def emergency_triage(
        self, 
        alert: Dict[str, Any], 
        patient_history: Dict[str, Any],
        available_staff: List[str]
    ) -> ClaudeResponse:
        """Emergency triage processing."""
        triage_response = f"""
EMERGENCY TRIAGE ASSESSMENT
Alert: {alert.get('type', 'Unknown')}
Priority: HIGH
Recommended Actions:
1. Immediate medical attention required
2. Contact available staff: {', '.join(available_staff[:3])}
3. Monitor vital signs continuously
4. Document all observations

This is an automated triage response. Human medical professional assessment required.
"""
        
        return ClaudeResponse(
            content=triage_response,
            reasoning="Emergency protocol activated",
            confidence=1.0,
            safety_flags=["emergency", "requires_medical_attention"],
            processing_time_ms=0.0,
            tokens_used=0
        )
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        self.logger.info("Conversation history cleared")