"""
FIN.DR — Voice Verification Conversation Engine (server-side state only)

This module implements a lightweight, session-based conversation engine
that contains the verification logic and stores per-transaction conversation
state. All microphone and TTS handling must happen in the browser; this
module only drives the question/response flow and persists results to the DB.

Enhanced with Ollama LLM integration for more human-like, natural responses.
"""
import re
import threading
from typing import Dict
import sys
import os
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VOICE_MAX_RETRIES, OLLAMA_HOST, OLLAMA_MODEL, USE_OLLAMA
from database.db import insert_voice_result

# Phrase sets
POSITIVE_WORDS = {"yes", "yeah", "yep", "confirm", "confirmed", "sure", "okay", "ok", "affirmative", "correct", "indeed", "absolutely", "definitely","s"}
POSITIVE_PHRASES = ["i did", "i authorized", "it was me", "that was me", "i made", "i confirm", "yes i did"]

NEGATIVE_WORDS = {"no", "nope", "never", "fraud", "unauthorized", "stop", "block"}
NEGATIVE_PHRASES = ["not me", "not mine", "didn't do", "did not do", "i did not", "unauthorized", "fraud", "someone else"]


def _call_ollama(prompt: str, timeout: int = 8) -> str:
    """Call Ollama API for human-like response generation. Fallback to default on failure."""
    if not USE_OLLAMA:
        return None
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.ok:
            return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"[VOICE-LLM] Ollama unavailable: {e}")
    return None


def _generate_verification_question(amount: float, merchant: str, timestamp: str) -> str:
    """Generate a human-like initial verification question using Ollama, with fallback."""
    llm_question = None
    if USE_OLLAMA:
        prompt = f"""You are a friendly bank fraud detection agent calling a customer.
        
Generate a natural, conversational question (2-3 sentences) to verify a transaction:
- Amount: ₹{amount}
- Merchant: {merchant}
- Timestamp: {timestamp}

The question should sound like a real person, not a robot. Ask if they authorized this transaction.
Keep it brief and natural. Respond with ONLY the question, no explanations."""
        
        llm_question = _call_ollama(prompt, timeout=8)
    
    if llm_question:
        return llm_question
    
    # Fallback to default
    return (
        f"We detected a transaction of ₹{amount} at {merchant} on {timestamp}. "
        f"Did you authorize this transaction?"
    )


def _generate_followup_question(attempt: int = 1) -> str:
    """Generate a human-like follow-up question using Ollama, with fallback."""
    llm_question = None
    if USE_OLLAMA:
        prompt = f"""You are a friendly bank fraud detection agent calling a customer.
        
The customer said 'no' to a transaction they were asked about. Now you need to ask a follow-up question
to confirm whether anyone else they authorized could have made this transaction.

Generate a natural, conversational follow-up question (1-2 sentences) that doesn't sound robotic.
Keep it brief and friendly. Respond with ONLY the question, no explanations."""
        
        llm_question = _call_ollama(prompt, timeout=8)
    
    if llm_question:
        return llm_question
    
    # Fallback
    return "Was anyone else authorized to use your account for this transaction?"


def _classify_response_with_llm(text: str) -> str:
    """Use Ollama to classify response sentiment (positive/negative/unclear) for better understanding."""
    if not USE_OLLAMA or not text or text in ("__timeout__", "__skip__"):
        return classify_response(text)
    
    try:
        prompt = f"""Classify this customer response as positive (yes/authorized), negative (no/unauthorized), or unclear.

Customer said: "{text}"

You are a fraud analyst. Be strict but fair. If they clearly said yes/confirmed, answer "positive".
If they clearly said no/denied, answer "negative". Otherwise answer "unclear".

Respond with ONLY one word: positive, negative, or unclear."""
        
        llm_result = _call_ollama(prompt, timeout=6)
        if llm_result:
            llm_result = llm_result.lower().strip()
            if "positive" in llm_result:
                return "positive"
            elif "negative" in llm_result:
                return "negative"
    except Exception as e:
        print(f"[VOICE-LLM-CLASSIFY] Error: {e}")
    
    # Fallback to rule-based classification
    return classify_response(text)


def classify_response(text: str) -> str:
    """Return one of: 'positive', 'negative', 'unclear'."""
    if not text or text in ("__timeout__", "__skip__"):
        return "unclear"
    t = text.lower().strip()
    
    # Check for positive phrases first
    for phrase in POSITIVE_PHRASES:
        if phrase in t:
            return "positive"
    
    # Check for positive words (with word boundary for most, but also check single "yes")
    for w in POSITIVE_WORDS:
        if w == "yes":
            # Handle "yes" with or without punctuation
            if re.search(rf"\b{re.escape(w)}\b|^{re.escape(w)}$", t):
                return "positive"
        else:
            if re.search(rf"\b{re.escape(w)}\b", t):
                return "positive"
    
    # Check for negative phrases
    for phrase in NEGATIVE_PHRASES:
        if phrase in t:
            return "negative"
    
    # Check for negative words
    for w in NEGATIVE_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return "negative"
    
    return "unclear"


class ConversationSession:
    def __init__(self, txn_id: str, amount: float, merchant: str, timestamp: str, max_retries: int = VOICE_MAX_RETRIES):
        self.txn_id = txn_id
        self.amount = float(amount)
        self.merchant = merchant
        self.timestamp = timestamp
        self.max_retries = int(max_retries)
        self.attempts = 0
        self.state = "QUESTION"  # QUESTION, PROCESSING, COMPLETE
        self.current_question = self._build_initial_question()
        self.result = None
        self.question_phase = "initial"  # Track which question phase we're in: 'initial' or 'followup'

    def _build_initial_question(self) -> str:
        """Build initial question using Ollama for natural language."""
        return _generate_verification_question(self.amount, self.merchant, self.timestamp)

    def _build_followup_question(self) -> str:
        """Build follow-up question using Ollama for natural language."""
        return _generate_followup_question()

    def handle_response(self, text: str) -> Dict:
        """Process a single response and return next action dict.

        Possible return shapes:
          - { 'phase': 'QUESTION', 'question': '...' }
          - { 'phase': 'COMPLETE', 'status': 'APPROVED'|'BLOCKED'|'FRAUD_REVIEW', ... }
        """
        self.state = "PROCESSING"
        self.attempts += 1
        # Use LLM-enhanced classification for better understanding
        sentiment = _classify_response_with_llm(text)

        if sentiment == "positive":
            res = {
                "phase": "COMPLETE",
                "status": "APPROVED",
                "reason": "Customer confirmed the transaction via voice verification.",
                "followup": "",
                "raw_response": text,
            }
            self._finalise(res)
            return res

        if sentiment == "negative":
            # If this is the initial question and they said no, ask follow-up
            if self.question_phase == "initial":
                self.question_phase = "followup"
                self.current_question = self._build_followup_question()
                self.state = "QUESTION"
                return {"phase": "QUESTION", "question": self.current_question}
            else:
                # User said no to follow-up question - block the transaction
                res = {
                    "phase": "COMPLETE",
                    "status": "BLOCKED",
                    "reason": "User denied transaction and no authorised person confirmed.",
                    "followup": "Transaction flagged as unauthorised fraud.",
                    "raw_response": text,
                }
                self._finalise(res)
                return res

        # unclear - generate a natural follow-up prompt using Ollama
        if self.attempts < self.max_retries:
            if USE_OLLAMA:
                llm_prompt = _call_ollama(
                    f"""The customer's response "{text}" was unclear in a fraud verification call.
Generate a brief, friendly re-prompt asking them to say yes or no more clearly.
Keep it to 1-2 sentences and friendly in tone.
Respond with ONLY the re-prompt, no explanations.""",
                    timeout=6
                )
                if llm_prompt:
                    self.current_question = llm_prompt
                else:
                    self.current_question = "I didn't catch that. Could you please say yes or no?"
            else:
                self.current_question = "I didn't catch that. Could you please say yes or no?"
            
            self.state = "QUESTION"
            return {"phase": "QUESTION", "question": self.current_question}
        else:
            res = {
                "phase": "COMPLETE",
                "status": "FRAUD_REVIEW",
                "reason": "Could not obtain a clear response after multiple attempts.",
                "followup": "Manual review required.",
                "raw_response": text,
            }
            self._finalise(res)
            return res

    def _finalise(self, result: Dict):
        self.state = "COMPLETE"
        self.result = result
        # Persist to DB (voice_verifications table)
        try:
            insert_voice_result(self.txn_id, result)
        except Exception:
            pass


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, ConversationSession] = {}
        self._lock = threading.Lock()

    def start_session(self, txn_id: str, amount: float, merchant: str, timestamp: str) -> Dict:
        with self._lock:
            if txn_id in self._sessions and self._sessions[txn_id].state != "COMPLETE":
                sess = self._sessions[txn_id]
            else:
                sess = ConversationSession(txn_id, amount, merchant, timestamp)
                self._sessions[txn_id] = sess
        return {"phase": "QUESTION", "question": sess.current_question}

    def handle_response(self, txn_id: str, text: str) -> Dict:
        with self._lock:
            sess = self._sessions.get(txn_id)
            if not sess:
                return {"phase": "COMPLETE", "status": "ERROR", "reason": "No active session", "raw_response": text}
            result = sess.handle_response(text)
            # If complete, optionally remove session (keep for a short time)
            if sess.state == "COMPLETE":
                # we keep session instance for history but it's finalised
                pass
            return result


# Module-level manager
manager = SessionManager()


def start_session(txn_id: str, amount: float, merchant: str, timestamp: str) -> Dict:
    return manager.start_session(txn_id, amount, merchant, timestamp)


def respond_to_session(txn_id: str, text: str) -> Dict:
    return manager.handle_response(txn_id, text)


if __name__ == "__main__":
    # quick manual test
    s = start_session("TXN123", 4999, "Amazon India", "2026-06-10 10:00")
    print(s)
