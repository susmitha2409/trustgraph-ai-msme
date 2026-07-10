"""
================================================================
TrustGraph AI — Dynamic Financial DNA Engine for MSMEs
api_manager.py

Purpose:
    Manages FOUR Groq API keys with automatic round-robin rotation
    and automatic failover. If a key fails due to rate limiting
    (429), quota exhaustion, timeout, or connection error, the
    manager silently switches to the next available key and retries.

    The calling code (llm_report.py) never needs to know which key
    was used or whether any key failed — it simply gets a response
    or a clearly handled final failure after all keys/retries are
    exhausted.
================================================================
"""

import os
import time
from groq import Groq
from groq import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
)


class APIManager:
    """
    Handles round-robin rotation and automatic failover across four
    Groq API keys, so the rest of the application never has to deal
    with individual key failures.

    Usage:
        manager = APIManager()
        response_text = manager.generate_completion(
            messages=[{"role": "user", "content": "Hello"}]
        )
    """

    # Environment variable names for the four Groq API keys
    KEY_ENV_NAMES = [
        "GROQ_API_KEY_1",
        "GROQ_API_KEY_2",
        "GROQ_API_KEY_3",
        "GROQ_API_KEY_4",
    ]

    # Default model used for report generation
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    # Max retries PER KEY before moving to the next key
    MAX_RETRIES_PER_KEY = 2

    # Seconds to wait between retries (simple fixed backoff, kept simple
    # on purpose for hackathon reliability/predictability)
    RETRY_DELAY_SECONDS = 1.5

    def __init__(self):
        """
        Initializes the API Manager by loading all four Groq API keys
        from environment variables and setting up round-robin state.
        """
        self._keys = self._load_keys()
        self._num_keys = len(self._keys)
        self._current_index = 0
        # Tracks keys that have failed within the current call cycle
        # so we don't retry a dead key twice in the same request.
        self._disabled_keys = set()

    # ------------------------------------------------------------------
    # STEP 1: KEY LOADING
    # ------------------------------------------------------------------
    def _load_keys(self) -> list:
        """
        Loads all configured Groq API keys from environment variables.
        Keys that are missing or empty are skipped silently.

        Returns:
            list: List of valid (non-empty) API key strings.
        """
        keys = []
        for env_name in self.KEY_ENV_NAMES:
            key_value = os.environ.get(env_name, "").strip()
            if key_value:
                keys.append(key_value)
        return keys

    def has_valid_keys(self) -> bool:
        """
        Checks whether at least one valid Groq API key is configured.

        Returns:
            bool: True if at least one key is available.
        """
        return self._num_keys > 0

    # ------------------------------------------------------------------
    # STEP 2: ROUND-ROBIN KEY SELECTION
    # ------------------------------------------------------------------
    def _get_next_key_index(self) -> int:
        """
        Returns the next key index in round-robin order, skipping any
        keys that have already been disabled during this call cycle.

        Returns:
            int: Index of the next usable key, or -1 if none remain.
        """
        for _ in range(self._num_keys):
            index = self._current_index
            self._current_index = (self._current_index + 1) % self._num_keys
            if index not in self._disabled_keys:
                return index
        return -1  # all keys disabled

    # ------------------------------------------------------------------
    # STEP 3: ERROR CLASSIFICATION
    # ------------------------------------------------------------------
    def _is_recoverable_error(self, error: Exception) -> bool:
        """
        Determines whether an error should trigger a key switch
        (rate limit, quota, timeout, connection issue) versus being
        a fatal/unexpected error.

        Args:
            error (Exception): The exception raised by the Groq SDK.

        Returns:
            bool: True if the error is recoverable via key rotation.
        """
        if isinstance(error, (RateLimitError, APITimeoutError, APIConnectionError)):
            return True

        if isinstance(error, APIStatusError):
            # 429 = rate limit / quota exceeded, 5xx = server-side issues
            if error.status_code == 429 or 500 <= error.status_code < 600:
                return True

        # Fallback: inspect the error message for common recoverable phrases
        error_text = str(error).lower()
        recoverable_phrases = [
            "429", "quota", "rate limit", "timeout", "connection",
            "temporarily unavailable", "service unavailable"
        ]
        return any(phrase in error_text for phrase in recoverable_phrases)

    # ------------------------------------------------------------------
    # STEP 4: CORE COMPLETION METHOD WITH FAILOVER
    # ------------------------------------------------------------------
    def generate_completion(
        self,
        messages: list,
        model: str = None,
        temperature: float = 0.4,
        max_tokens: int = 1200,
    ) -> str:
        """
        Sends a chat completion request to Groq, automatically rotating
        and retrying across all configured API keys on failure. Never
        raises to the caller under normal failure modes — returns a
        graceful fallback message instead if all keys are exhausted.

        Args:
            messages (list): List of chat messages, e.g.
                              [{"role": "user", "content": "..."}]
            model (str): Groq model name (defaults to DEFAULT_MODEL).
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens in the response.

        Returns:
            str: The generated text response, or a graceful fallback
                 message if every key/retry attempt failed.
        """
        if not self.has_valid_keys():
            return (
                "⚠️ AI Report unavailable: No Groq API keys configured. "
                "Please add GROQ_API_KEY_1 through GROQ_API_KEY_4 to continue."
            )

        model = model or self.DEFAULT_MODEL
        self._disabled_keys = set()  # reset failure tracking for this call

        attempts_log = []  # internal-only, never shown to the user

        while len(self._disabled_keys) < self._num_keys:
            key_index = self._get_next_key_index()
            if key_index == -1:
                break  # all keys disabled

            api_key = self._keys[key_index]
            client = Groq(api_key=api_key)

            for retry_num in range(self.MAX_RETRIES_PER_KEY):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content.strip()

                except Exception as error:
                    attempts_log.append(
                        f"key_index={key_index} retry={retry_num} error={type(error).__name__}"
                    )

                    if self._is_recoverable_error(error):
                        if retry_num < self.MAX_RETRIES_PER_KEY - 1:
                            time.sleep(self.RETRY_DELAY_SECONDS)
                            continue  # retry same key
                        else:
                            self._disabled_keys.add(key_index)
                            break  # move to next key
                    else:
                        # Non-recoverable / unexpected error — still fail
                        # gracefully rather than crashing the app.
                        self._disabled_keys.add(key_index)
                        break

        # All keys exhausted — return a safe fallback, never crash the app
        return (
            "⚠️ AI Report temporarily unavailable. All configured Groq API "
            "keys are currently rate-limited or unreachable. Please try "
            "again in a moment."
        )

    # ------------------------------------------------------------------
    # STEP 5: STATUS UTILITY (for internal diagnostics, not shown to user
    #          unless explicitly requested — keeps key identities hidden)
    # ------------------------------------------------------------------
    def get_status_summary(self) -> dict:
        """
        Returns a high-level, non-sensitive status summary of the
        API Manager (number of keys configured, none of the actual
        key values or which specific key is active).

        Returns:
            dict: Status summary safe to display in the UI.
        """
        return {
            "keys_configured": self._num_keys,
            "keys_available": self._num_keys - len(self._disabled_keys),
            "rotation_mode": "round_robin",
        }


# ------------------------------------------------------------------
# STANDALONE TEST (only runs if this file is executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    manager = APIManager()
    print("Status:", manager.get_status_summary())

    if manager.has_valid_keys():
        reply = manager.generate_completion(
            messages=[{"role": "user", "content": "Say hello in one short sentence."}]
        )
        print("Response:", reply)
    else:
        print("No keys configured — add GROQ_API_KEY_1..4 to your .env file.")
