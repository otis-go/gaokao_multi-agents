from __future__ import annotations

"""
src/shared/llm_interface.py

LLMClient is a lightweight wrapper for underlying LLM calls:
- GenerationOrchestrator and EvaluationOrchestrator only interact with this class
- Switching between OpenAI or other providers can be done here without affecting Agent code

Current version only has built-in support for `api_type="openai"`,
other types (like "dummy" / "offline") return placeholder content for local debugging.

Multi-model Parallel Call Notes:
- batch_call: Supports concurrent calls to multiple models, returns {model_name: response} dict
- Uses concurrent.futures.ThreadPoolExecutor for parallelism
- Stage 2 evaluation uses this for multi-model ensemble evaluation

Logging Features:
- Auto-logs input prompts and output responses for each LLM call
- Uses global logger from llm_logger module
- Can be enabled/disabled via init_global_logger()

[2025-12] Auto-retry Mechanism:
- Default 3 retries for all API calls
- Supports auto-recovery from network fluctuations, API timeouts, empty responses
- Retry info is recorded in audit log

[2025-12] Network Disconnection Recovery:
- Detects network connection errors (Connection Error, Timeout, etc.)
- Uses exponential backoff strategy when network is down
- Default: infinite wait until network recovers (configurable)
- Probes network connectivity every 10 seconds while waiting
- Retries immediately after network recovery, no need to wait full delay cycle
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import os
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# [2025-12] Import model capability check functions
from src.shared.api_config import (
    is_no_temperature_model,
    clamp_max_tokens,
    is_no_max_tokens_model,
    is_reasoning_effort_model,
    is_ark_model,
)

# Import logger
try:
    from src.shared.llm_logger import get_global_logger, log_llm_call
except ImportError:
    get_global_logger = lambda: None
    log_llm_call = lambda **kwargs: None

try:
    # Compatible with old openai==0.x and new openai>=1.x
    import openai  # type: ignore
    # Check openai version
    _openai_version = getattr(openai, "__version__", "0.0.0")
    _openai_major_version = int(_openai_version.split(".")[0])
    _use_new_api = _openai_major_version >= 1
except ImportError:  # Fallback if openai not installed
    openai = None  # type: ignore
    _use_new_api = False

# Google Gemini SDK (optional dependency)
try:
    from google import genai as google_genai  # type: ignore
    _google_genai_available = True
except ImportError:
    google_genai = None  # type: ignore
    _google_genai_available = False


# =============================================================================
# [2025-12] Global Retry Audit Log
# Collects all retry and failure info across calls
# =============================================================================

class LLMRetryAudit:
    """
    LLM call retry audit logger

    Collects all LLM call failures and retries in current experiment/session,
    for displaying network fluctuations etc. in final report.
    """

    def __init__(self):
        self.retry_records: List[Dict[str, Any]] = []
        self.failure_records: List[Dict[str, Any]] = []
        self._enabled = True

    def record_retry(
        self,
        model_name: str,
        attempt: int,
        max_attempts: int,
        error_msg: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a retry
        """
        if not self._enabled:
            return
        self.retry_records.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "error_msg": error_msg,
            "context": context or {},
        })

    def record_failure(
        self,
        model_name: str,
        total_attempts: int,
        errors: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record final failure (all retries exhausted)
        """
        if not self._enabled:
            return
        self.failure_records.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "total_attempts": total_attempts,
            "errors": errors,
            "context": context or {},
        })

    def get_summary(self) -> Dict[str, Any]:
        """
        Get audit summary
        """
        return {
            "total_retries": len(self.retry_records),
            "total_failures": len(self.failure_records),
            "retry_records": self.retry_records,
            "failure_records": self.failure_records,
        }

    def has_issues(self) -> bool:
        """
        Check if there are issues (retries or failures)
        """
        return len(self.retry_records) > 0 or len(self.failure_records) > 0

    def clear(self) -> None:
        """
        Clear records (call when starting new experiment)
        """
        self.retry_records.clear()
        self.failure_records.clear()


# Global audit logger instance
_global_retry_audit = LLMRetryAudit()


def get_retry_audit() -> LLMRetryAudit:
    """
    Get global retry audit logger
    """
    return _global_retry_audit


def clear_retry_audit() -> None:
    """
    Clear global retry audit records
    """
    _global_retry_audit.clear()


# =============================================================================
# [2025-12] Network Disconnection Detection & Recovery
# =============================================================================

# Keywords for network-related errors (for detecting network issues)
NETWORK_ERROR_KEYWORDS = [
    "connection",
    "timeout",
    "timed out",
    "network",
    "socket",
    "refused",
    "reset",
    "unreachable",
    "dns",
    "name resolution",
    "ssl",
    "certificate",
    "proxy",
    "gateway",
    "502",
    "503",
    "504",
    "429",  # Rate limit also treated as network issue
    "rate limit",
    "too many requests",
]


def _is_network_error(error_msg: str) -> bool:
    """
    Check if error is network-related.

    Args:
        error_msg: Error message string

    Returns:
        True if network-related error, False otherwise
    """
    if not error_msg:
        return False
    error_lower = error_msg.lower()
    return any(kw in error_lower for kw in NETWORK_ERROR_KEYWORDS)


def _wait_for_network_recovery(
    model_name: str,
    error_msg: str,
    attempt: int,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
    max_total_wait: float = 300.0,  # Max total wait 5 minutes
    infinite_wait: bool = False,
) -> bool:
    """
    Wait for network recovery using exponential backoff strategy.

    Args:
        model_name: Model name
        error_msg: Error message
        attempt: Current attempt number
        base_delay: Base delay in seconds
        max_delay: Max single delay in seconds
        max_total_wait: Max total wait time (seconds)
        infinite_wait: Whether to wait infinitely

    Returns:
        True if should continue retrying, False if should give up
    """
    # Calculate exponential backoff delay: base_delay * 2^(attempt-1)
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

    print(f"\n[Network Recovery] Detected network error: {error_msg[:100]}...")
    print(f"[Network Recovery] {model_name} attempt {attempt}, waiting {delay:.1f} seconds before retry...")

    # Record to audit log
    _global_retry_audit.record_retry(
        model_name=model_name,
        attempt=attempt,
        max_attempts=-1 if infinite_wait else int(max_total_wait / base_delay),
        error_msg=f"[Network recovery wait] {error_msg}",
        context={"wait_seconds": delay, "network_error": True},
    )

    time.sleep(delay)
    return True


def _test_network_connectivity(test_urls: List[str] = None) -> bool:
    """
    Test network connectivity.

    Checks if network has recovered by attempting to connect to common test endpoints.

    Args:
        test_urls: Test URL list, defaults to common API endpoints

    Returns:
        True if network has recovered, False otherwise
    """
    import socket
    import urllib.request

    if test_urls is None:
        # Default test endpoints: using common, stable API endpoints
        test_urls = [
            "https://api.openai.com",
            "https://api.deepseek.com",
            "https://www.dmxapi.cn",
            "https://www.baidu.com",
        ]

    for url in test_urls:
        try:
            # Extract hostname
            if url.startswith("https://"):
                host = url[8:].split("/")[0]
            elif url.startswith("http://"):
                host = url[7:].split("/")[0]
            else:
                host = url.split("/")[0]

            # Try DNS resolution and TCP connection
            socket.create_connection((host, 443), timeout=5)
            return True
        except (socket.timeout, socket.error, OSError):
            continue

    return False


class NetworkRecoveryConfig:
    """
    Network recovery configuration class.

    Allows runtime configuration of network recovery behavior.

    [2025-12 Improvements]
    - Default: infinite wait until network recovers
    - Proactively probes network connectivity while waiting
    - Retries immediately after network recovery, no need to wait full delay cycle
    """

    def __init__(
        self,
        enabled: bool = True,
        base_delay: float = 5.0,
        max_delay: float = 60.0,
        max_total_wait: float = 3600.0,  # Default 1 hour
        infinite_wait: bool = True,  # Default infinite wait
        probe_interval: float = 10.0,  # Network probe interval (seconds)
    ):
        self.enabled = enabled
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_total_wait = max_total_wait
        self.infinite_wait = infinite_wait
        self.probe_interval = probe_interval
        self._total_wait_time = 0.0

    def should_retry(self, error_msg: str) -> bool:
        """
        Check if should retry due to network error
        """
        if not self.enabled:
            return False
        return _is_network_error(error_msg)

    def wait_and_check(self, model_name: str, error_msg: str, attempt: int) -> bool:
        """
        Wait and check if should continue retrying.

        [Improvement] Proactively probes network while waiting, returns immediately when recovered.

        Returns:
            True if should continue retrying, False if should give up
        """
        if not self.enabled:
            return False

        # Check if exceeded max total wait time
        if not self.infinite_wait and self._total_wait_time >= self.max_total_wait:
            print(f"[Network Recovery] Reached max wait time {self.max_total_wait:.0f}s, giving up retry")
            return False

        # Calculate this round's wait time (exponential backoff, capped)
        wait_time = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)

        # Check if this wait would exceed max total wait time
        if not self.infinite_wait and (self._total_wait_time + wait_time) > self.max_total_wait:
            wait_time = self.max_total_wait - self._total_wait_time

        if wait_time <= 0:
            return False

        print(f"\n[Network Recovery] Detected network error: {error_msg[:100]}...")
        print(f"[Network Recovery] {model_name} starting network recovery wait...")
        if self.infinite_wait:
            print(f"[Network Recovery] Mode: infinite wait, probing network every {self.probe_interval:.0f} seconds")
        else:
            print(f"[Network Recovery] Waited {self._total_wait_time:.0f}s / max {self.max_total_wait:.0f}s")

        # Record to audit log
        _global_retry_audit.record_retry(
            model_name=model_name,
            attempt=attempt,
            max_attempts=-1 if self.infinite_wait else int((self.max_total_wait - self._total_wait_time) / self.base_delay),
            error_msg=f"[Network recovery wait] {error_msg}",
            context={
                "wait_seconds": wait_time,
                "network_error": True,
                "total_wait_so_far": self._total_wait_time,
            },
        )

        # [Improvement] Segmented waiting, proactively probes network during wait
        waited = 0.0
        probe_count = 0
        while waited < wait_time:
            # Calculate this sleep duration
            sleep_time = min(self.probe_interval, wait_time - waited)
            time.sleep(sleep_time)
            waited += sleep_time
            self._total_wait_time += sleep_time

            # Proactively probe network connectivity
            probe_count += 1
            if _test_network_connectivity():
                print(f"[Network Recovery] [OK] Probe {probe_count} succeeded! Network recovered, retrying immediately...")
                return True
            else:
                # Display wait progress
                if self.infinite_wait:
                    print(f"[Network Recovery] Probe {probe_count}, network still down, waited {self._total_wait_time:.0f} seconds...")
                else:
                    remaining = self.max_total_wait - self._total_wait_time
                    print(f"[Network Recovery] Probe {probe_count}, network still down, remaining {remaining:.0f} seconds...")

        # Wait time completed, ready to retry
        print(f"[Network Recovery] Wait {wait_time:.1f} seconds completed, attempting to call again...")
        return True

    def reset(self):
        """
        Reset wait time counter (called at start of each LLM call)
        """
        self._total_wait_time = 0.0


# Global network recovery configuration
_global_network_config = NetworkRecoveryConfig()


def get_network_config() -> NetworkRecoveryConfig:
    """
    Get global network recovery configuration
    """
    return _global_network_config


def set_network_config(
    enabled: bool = True,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
    max_total_wait: float = 3600.0,
    infinite_wait: bool = True,
    probe_interval: float = 10.0,
) -> None:
    """
    Set global network recovery configuration.

    Args:
        enabled: Whether to enable network recovery
        base_delay: Base delay in seconds
        max_delay: Max single delay in seconds
        max_total_wait: Max total wait (seconds), default 1 hour
        infinite_wait: Whether to wait infinitely (default True)
        probe_interval: Network probe interval in seconds (default 10s)
    """
    global _global_network_config
    _global_network_config = NetworkRecoveryConfig(
        enabled=enabled,
        base_delay=base_delay,
        max_delay=max_delay,
        max_total_wait=max_total_wait,
        infinite_wait=infinite_wait,
        probe_interval=probe_interval,
    )


@dataclass
class LLMClient:
    """
    LLM client wrapper class

    Wraps calls to various LLM APIs, supporting OpenAI-compatible and Google Gemini.
    """
    api_type: str
    model_name: str
    verbose: bool = False
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    api_key: Optional[str] = None  # Direct API key (priority over env var)
    # [2025-12] Store instance-level max_tokens and temperature
    max_tokens: int = 16384  # Default 16384, avoid Gemini truncation
    # [2025-12-26] temperature default aligned with api_config.py
    temperature: Optional[float] = 1.0  # Stage1 default temp, can be overridden by llm_router

    def __post_init__(self) -> None:
        # Priority: direct api_key > environment variable
        if not self.api_key:
            self.api_key = os.getenv(self.api_key_env)
        self._google_client = None  # Google Gemini client cache

        # [2025-12] Self-check suite mock mode support
        # When SELF_CHECK_NO_LLM=1, force dummy mode
        if os.getenv("SELF_CHECK_NO_LLM") == "1":
            self.api_type = "dummy"
            print(f"[LLMClient] SELF_CHECK_NO_LLM=1, forcing dummy mode (model={self.model_name})")

        if self.verbose:
            print(
                f"[LLMClient] Initialize: api_type={self.api_type}, "
                f"model_name={self.model_name}, api_key_env={self.api_key_env}"
            )

        if self.api_type == "openai":
            if openai is None:
                raise ImportError(
                    "openai library not installed, but LLMConfig.api_type is 'openai'.\n"
                    "Please run `pip install openai` or switch to another backend."
                )
            if not self.api_key:
                raise ValueError(
                    f"API key not configured, cannot call OpenAI-compatible API (model={self.model_name}).\n"
                    f"Please configure api_key in src/shared/api_config.py."
                )

            # Old openai==0.x version
            if not _use_new_api:
                openai.api_key = self.api_key  # type: ignore[attr-defined]
                if self.base_url:
                    openai.base_url = self.base_url  # type: ignore[attr-defined]

        elif self.api_type == "google_genai":
            # Google Gemini API
            if not _google_genai_available:
                raise ImportError(
                    "google-genai library not installed, but api_type is 'google_genai'.\n"
                    "Please run `pip install google-genai` or switch to another backend."
                )
            if not self.api_key:
                raise ValueError(
                    f"API key not configured, cannot call Google Gemini API (model={self.model_name}).\n"
                    f"Please configure api_key in src/shared/api_config.py."
                )
            # Initialize Google Gemini client
            self._google_client = google_genai.Client(api_key=self.api_key)
            if self.verbose:
                print(f"[LLMClient] Google Gemini client initialized: model={self.model_name}")

        elif self.api_type in ("dummy", "offline"):
            # Offline / debug mode, no extra initialization
            if self.verbose:
                print("[LLMClient] Using dummy/offline mode, returns placeholder responses only.")
        else:
            # You can extend here for other providers (e.g., Alibaba, Zhipu)
            raise ValueError(f"Unsupported api_type: {self.api_type!r}")

    # ------------------------------------------------------------------
    # Public unified call interface
    # ------------------------------------------------------------------
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """
        Unified call entry point for upper-layer Agents.

        [2025-12] Auto-retry mechanism:
        - Default max 3 retries
        - Network errors, API timeouts, empty responses trigger retry
        - Retry info recorded in global audit log

        [2025-12] Network disconnection recovery:
        - Uses exponential backoff on network errors
        - Network recovery wait independent of retry count
        - Max wait configurable via set_network_config()

        [2025-12 Improvement] Uses instance-configured max_tokens and temperature by default
        - If not specified at call time, uses self.max_tokens and self.temperature
        - Avoids manual parameter passing everywhere

        Args:
            messages: OpenAI-style message list:
                [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
            temperature: Sampling temp (defaults to instance config)
            max_tokens: Max tokens (defaults to instance config)
            metadata: Extra metadata for logging (e.g., agent name, unit_id)
            max_retries: Max retries (default 3)
            retry_delay: Retry delay in seconds (default 1)
            **kwargs: Pass-through params for underlying SDK

        Returns:
            Text content returned by LLM (first choice only).
        """
        # [2025-12 Improvement] Use instance-configured defaults
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        errors_collected: List[str] = []
        content = ""

        # Get network recovery config and reset wait time
        network_config = get_network_config()
        network_config.reset()

        # Network recovery retry counter (independent of normal retries)
        network_retry_count = 0

        attempt = 0
        while attempt < max_retries:
            attempt += 1
            start_time = time.time()
            error_msg = None

            try:
                content = self._do_generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                # Check if empty response (also treated as failure, needs retry)
                if not content or not content.strip():
                    error_msg = "Model returned empty response"
                    errors_collected.append(f"[Attempt {attempt}] {error_msg}")

                    if attempt < max_retries:
                        # Record retry
                        _global_retry_audit.record_retry(
                            model_name=self.model_name,
                            attempt=attempt,
                            max_attempts=max_retries,
                            error_msg=error_msg,
                            context=metadata,
                        )
                        print(f"[LLMClient] {self.model_name} returned empty response, attempt {attempt}/{max_retries} failed, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Last attempt also failed
                        _global_retry_audit.record_failure(
                            model_name=self.model_name,
                            total_attempts=max_retries,
                            errors=errors_collected,
                            context=metadata,
                        )
                        print(f"[LLMClient] {self.model_name} returned empty response {max_retries} times, giving up retry")
                        # Don't throw exception, return empty string for upper layer to handle
                        break

                # Successfully got content
                if attempt > 1 or network_retry_count > 0:
                    print(f"[LLMClient] {self.model_name} attempt {attempt} succeeded" +
                          (f" (after {network_retry_count} network recovery waits)" if network_retry_count > 0 else ""))

                # Record call log
                latency_ms = (time.time() - start_time) * 1000
                log_llm_call(
                    model_name=self.model_name,
                    messages=messages,
                    response=content,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    latency_ms=latency_ms,
                    metadata={
                        **(metadata or {}),
                        "attempt": attempt,
                        "retries_before_success": attempt - 1,
                        "network_retries": network_retry_count,
                    },
                    error=None,
                )

                if self.verbose:
                    print(
                        f"[LLMClient] Call complete: model={self.model_name}, "
                        f"temperature={temperature}, max_tokens={max_tokens}, "
                        f"response_len={len(content)}, latency={latency_ms:.2f}ms, "
                        f"attempt={attempt}/{max_retries}, network_retries={network_retry_count}"
                    )

                return content

            except Exception as e:
                error_msg = str(e)
                errors_collected.append(f"[Attempt {attempt}] {error_msg}")

                # Calculate latency and record error log
                latency_ms = (time.time() - start_time) * 1000
                log_llm_call(
                    model_name=self.model_name,
                    messages=messages,
                    response="",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    latency_ms=latency_ms,
                    metadata={
                        **(metadata or {}),
                        "attempt": attempt,
                    },
                    error=error_msg,
                )

                # [2025-12] Detect if network error, enable network recovery mechanism
                if network_config.should_retry(error_msg):
                    network_retry_count += 1
                    if network_config.wait_and_check(self.model_name, error_msg, network_retry_count):
                        # Network recovery wait succeeded, don't consume normal retry count
                        attempt -= 1  # Restore attempt count
                        continue

                if attempt < max_retries:
                    # Still have retry chances
                    _global_retry_audit.record_retry(
                        model_name=self.model_name,
                        attempt=attempt,
                        max_attempts=max_retries,
                        error_msg=error_msg,
                        context=metadata,
                    )
                    print(f"[LLMClient] {self.model_name} call failed: {error_msg}")
                    print(f"[LLMClient] Attempt {attempt}/{max_retries} failed, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    # All retries exhausted
                    _global_retry_audit.record_failure(
                        model_name=self.model_name,
                        total_attempts=max_retries,
                        errors=errors_collected,
                        context=metadata,
                    )
                    print(f"[LLMClient] {self.model_name} failed {max_retries} times, giving up retry")
                    print(f"[LLMClient] Error history: {errors_collected}")
                    raise  # Raise last exception

        return content

    def _do_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: int,
        **kwargs: Any,
    ) -> str:
        """
        Internal method that actually executes LLM call (no retry logic)
        """
        content = ""

        if self.api_type == "openai":
            assert openai is not None

            if not _use_new_api:
                # Old 0.x API
                # [2025-12] Adjust max_tokens based on model limits
                clamped_max_tokens = clamp_max_tokens(self.model_name, max_tokens)

                old_api_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": clamped_max_tokens,
                }
                if temperature is not None:
                    old_api_params["temperature"] = temperature
                old_api_params.update(kwargs)

                resp = openai.ChatCompletion.create(**old_api_params)
                content = resp["choices"][0]["message"]["content"]
            else:
                # New 1.x API
                from openai import OpenAI

                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )

                # [2025-12 Fix] New OpenAI models (gpt-5.1, o1, etc.) and Doubao Seed models
                # use max_completion_tokens instead of max_tokens parameter.
                # Reference: https://platform.openai.com/docs/api-reference/chat/create
                use_new_token_param = is_no_max_tokens_model(self.model_name)

                # [2025-12 Fix] Check if model doesn't support temperature parameter
                # gpt-5-mini, o1 series, doubao-seed etc. don't support temperature
                # Two cases to skip temperature:
                # 1. temperature is None (caller explicitly skips)
                # 2. Model doesn't support temp (checked by is_no_temperature_model)
                skip_temperature = temperature is None or is_no_temperature_model(self.model_name)

                # Build API call parameters
                api_params = {
                    "model": self.model_name,
                    "messages": messages,
                }

                # Only add temperature if supported
                if not skip_temperature:
                    api_params["temperature"] = temperature

                # [2025-12] Adjust max_tokens based on model limits
                # Some models (e.g., deepseek-chat) have max_tokens upper limit
                clamped_max_tokens = clamp_max_tokens(self.model_name, max_tokens)
                if clamped_max_tokens != max_tokens and self.verbose:
                    print(f"[LLMClient] max_tokens adjusted from {max_tokens} to {clamped_max_tokens} (model limit)")

                # Select token param based on model type
                if use_new_token_param:
                    api_params["max_completion_tokens"] = clamped_max_tokens
                else:
                    api_params["max_tokens"] = clamped_max_tokens

                # [2025-12] Doubao Seed etc. models support reasoning_effort parameter
                if is_reasoning_effort_model(self.model_name):
                    # Get reasoning_effort from kwargs, default "medium"
                    reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                    if reasoning_effort:
                        api_params["reasoning_effort"] = reasoning_effort
                        if self.verbose:
                            print(f"[LLMClient] Doubao reasoning model, set reasoning_effort={reasoning_effort}")

                # Add other kwargs
                api_params.update(kwargs)

                resp = client.chat.completions.create(**api_params)
                content = resp.choices[0].message.content or ""

                # DeepSeek-Reasoner special handling: check reasoning_content
                if not content:
                    msg = resp.choices[0].message
                    reasoning = getattr(msg, 'reasoning_content', None)
                    if not reasoning and hasattr(msg, 'model_extra'):
                        reasoning = msg.model_extra.get('reasoning_content', '')
                    if reasoning and self.verbose:
                        print(f"[LLMClient] Detected empty content, but has reasoning_content")

        elif self.api_type == "google_genai":
            assert self._google_client is not None

            # Convert OpenAI-style messages to Gemini format
            gemini_contents = self._convert_messages_to_gemini(messages)

            # [2025-12 Fix] Correct Gemini API call method
            # Reference: https://ai.google.dev/gemini-api/docs/text-generation
            #
            # [2025-12 Re-fix] Ensure max_output_tokens is always passed
            # Previous issue: when types import fails, falls back to simplified call, max_output_tokens not passed
            # Gemini default max_output_tokens is small, causing output truncation
            response = None
            gen_config_used = False

            try:
                # Method 1: Try using google.genai.types (new SDK >= 0.3)
                from google.genai import types
                # [2025-12] Support models without temperature parameter
                if temperature is not None:
                    gen_config = types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                else:
                    gen_config = types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                    )
                response = self._google_client.models.generate_content(
                    model=self.model_name,
                    contents=gemini_contents,
                    config=gen_config,
                )
                gen_config_used = True
                if self.verbose:
                    print(f"[LLMClient] Gemini: Using types.GenerateContentConfig, max_output_tokens={max_tokens}")
            except (ImportError, TypeError, AttributeError) as e1:
                if self.verbose:
                    print(f"[LLMClient] Gemini: types.GenerateContentConfig unavailable ({e1}), trying other method")

                try:
                    # Method 2: Try using generation_config dict (some versions support)
                    # [2025-12] Support models without temperature parameter
                    gen_cfg_dict = {"max_output_tokens": max_tokens}
                    if temperature is not None:
                        gen_cfg_dict["temperature"] = temperature
                    response = self._google_client.models.generate_content(
                        model=self.model_name,
                        contents=gemini_contents,
                        generation_config=gen_cfg_dict,
                    )
                    gen_config_used = True
                    if self.verbose:
                        print(f"[LLMClient] Gemini: Using generation_config dict, max_output_tokens={max_tokens}")
                except (TypeError, AttributeError) as e2:
                    if self.verbose:
                        print(f"[LLMClient] Gemini: generation_config dict unavailable ({e2}), using simplified call")

                    # Method 3: Last fallback - simplified call (cannot control max_output_tokens)
                    # Note: output may be truncated in this case!
                    print(f"[LLMClient] [Warning] Gemini cannot set max_output_tokens, output may be truncated!")
                    response = self._google_client.models.generate_content(
                        model=self.model_name,
                        contents=gemini_contents,
                    )

            # Extract response text
            # Prefer response.text, if unavailable extract from candidates
            try:
                content = response.text or ""
            except (AttributeError, ValueError) as e:
                # In some cases response.text may not be available
                # Try extracting from candidates
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            content = candidate.content.parts[0].text or ""
                        else:
                            content = ""
                    else:
                        content = ""
                else:
                    content = ""
                if self.verbose:
                    print(f"[LLMClient] Gemini response.text unavailable, extracting from candidates: {e}")

            # [2025-12 Diagnostic] Print Gemini response finish_reason, helps diagnose truncation
            # Possible finish_reason values:
            # - STOP: Normal completion (model considers answer complete)
            # - MAX_TOKENS: Truncated due to max_output_tokens limit
            # - SAFETY: Safety filter
            # - RECITATION: Citation issue
            # - OTHER: Other reason
            if self.verbose or not content:
                try:
                    finish_reason = None
                    prompt_token_count = None
                    candidates_token_count = None

                    # Try getting finish_reason
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'finish_reason'):
                            finish_reason = candidate.finish_reason

                    # Try getting token usage
                    if hasattr(response, 'usage_metadata'):
                        usage = response.usage_metadata
                        if hasattr(usage, 'prompt_token_count'):
                            prompt_token_count = usage.prompt_token_count
                        if hasattr(usage, 'candidates_token_count'):
                            candidates_token_count = usage.candidates_token_count

                    print(f"[LLMClient] Gemini response diagnostics:")
                    print(f"  - finish_reason: {finish_reason}")
                    print(f"  - prompt_tokens: {prompt_token_count}")
                    print(f"  - output_tokens: {candidates_token_count}")
                    print(f"  - content_length: {len(content)} chars")

                    # Diagnostic suggestions based on finish_reason
                    if finish_reason:
                        reason_str = str(finish_reason)
                        if 'MAX_TOKENS' in reason_str or 'LENGTH' in reason_str:
                            print(f"  [Diagnostic] Output truncated! Reason: reached max_output_tokens limit")
                            print(f"  [Suggestion] Increase max_tokens setting in api_config.py")
                        elif 'STOP' in reason_str:
                            print(f"  [Diagnostic] Model considers answer complete (STOP)")
                            if len(content) < 100:
                                print(f"  [Note] Output very short but model considers complete, may be model understanding or prompt issue")
                        elif 'SAFETY' in reason_str:
                            print(f"  [Diagnostic] Response blocked by safety filter")
                        elif 'RECITATION' in reason_str:
                            print(f"  [Diagnostic] Response blocked due to citation issue")
                        elif 'BLOCKED' in reason_str:
                            print(f"  [Diagnostic] Response blocked")
                except Exception as diag_e:
                    if self.verbose:
                        print(f"[LLMClient] Gemini diagnostic info fetch failed: {diag_e}")

        elif self.api_type in ("dummy", "offline"):
            # Mock response
            last_user = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user = msg.get("content", "")
                    break

            if "json" in last_user.lower() or "JSON" in last_user:
                content = '''{
    "overall_score": 75.0,
    "dimension_scores": {"accuracy": 80, "completeness": 70, "clarity": 75},
    "feedback": "[MOCK] This is a mock evaluation response from self-check suite",
    "pass": true
}'''
            elif "evaluate" in last_user.lower() or "score" in last_user.lower():
                content = '''{
    "overall_score": 75.0,
    "total_score": 75.0,
    "score": 75.0,
    "feedback": "[MOCK] Mock evaluation complete",
    "pass": true
}'''
            elif "single-choice" in last_user.lower():
                content = '''```json
{
    "stem": "[MOCK] This is a mock multiple-choice question stem",
    "question_type": "single-choice",
    "options": [
        {"label": "A", "content": "Mock option A"},
        {"label": "B", "content": "Mock option B"},
        {"label": "C", "content": "Mock option C"},
        {"label": "D", "content": "Mock option D"}
    ],
    "standard_answer": "A",
    "explanation": "[MOCK] This is mock explanation"
}
```'''
            else:
                content = (
                    f"[LLMClient dummy({self.model_name}) response, for debugging only]\n"
                    f"[MOCK] Mock response content\n"
                    f"Original input length: {len(last_user)} characters"
                )
        else:
            raise ValueError(f"Unsupported api_type: {self.api_type!r}")

        return content

    # ------------------------------------------------------------------
    # Google Gemini message format conversion
    # ------------------------------------------------------------------
    def _convert_messages_to_gemini(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to Gemini format.

        [2025-12 Improvement]
        Gemini API contents parameter supports:
        - Simple string (single-turn conversation, recommended for most cases)
        - Content object list (multi-turn conversation)

        For compatibility and simplicity, merge all messages into a single string here.
        System messages become context prefix, User and Assistant messages concatenated in order.

        Args:
            messages: OpenAI-style message list

        Returns:
            Converted Gemini contents string
        """
        # Separate system messages and conversation messages
        system_parts = []
        conversation_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if not content:
                continue

            if role == "system":
                # Collect system messages, put at front
                system_parts.append(content)
            elif role == "user":
                conversation_parts.append(f"User: {content}")
            elif role == "assistant":
                conversation_parts.append(f"Assistant: {content}")
            else:
                conversation_parts.append(f"{role}: {content}")

        # Build final prompt
        final_parts = []

        # System instructions go first
        if system_parts:
            system_text = "\n\n".join(system_parts)
            final_parts.append(f"[System Instructions]\n{system_text}\n")

        # If only one user message, return content directly (simplest format)
        if len(conversation_parts) == 1 and not system_parts:
            # Remove "User: " prefix
            return conversation_parts[0].replace("User: ", "", 1)

        # Multi-turn conversation or has system message, use structured format
        if conversation_parts:
            final_parts.append("[Conversation]\n" + "\n".join(conversation_parts))

        return "\n".join(final_parts)

    # ------------------------------------------------------------------
    # Multi-model parallel call interface (Stage 2 multi-model evaluation)
    # ------------------------------------------------------------------
    @staticmethod
    def batch_call(
        clients: List["LLMClient"],
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_workers: int = 5,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Parallel call to multiple LLMClients, returns each model's response.

        Used for Stage 2 multi-model ensemble evaluation, multiple models evaluate the same question simultaneously.

        Args:
            clients: LLMClient instance list
            messages: OpenAI-style message list (all models use same messages)
            temperature: Sampling temperature (Stage 2 default 0.0 for stability)
            max_tokens: Max tokens to generate
            max_workers: Max parallel thread count
            **kwargs: Pass-through params for underlying generate()

        Returns:
            Dict[str, str]: {model_name: response_content} dict
            If a model call fails, corresponding value is error message string
        """
        results: Dict[str, str] = {}

        def call_single(client: "LLMClient") -> tuple:
            """Call single model"""
            try:
                response = client.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return (client.model_name, response, None)
            except Exception as e:
                return (client.model_name, None, str(e))

        # Use thread pool for parallel calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(call_single, client): client for client in clients}

            for future in as_completed(futures):
                model_name, response, error = future.result()
                if error:
                    results[model_name] = f"[ERROR] {error}"
                    print(f"[LLMClient.batch_call] Model {model_name} call failed: {error}")
                else:
                    results[model_name] = response

        return results

    @staticmethod
    def create_clients_from_config(
        model_configs: List[Any],
        verbose: bool = False,
    ) -> List["LLMClient"]:
        """
        Create multiple LLMClient instances from config list.

        Args:
            model_configs: ModelWeightConfig list
            verbose: Whether to print verbose logs

        Returns:
            List[LLMClient]: LLMClient instance list
        """
        clients = []
        for config in model_configs:
            client = LLMClient(
                api_type=config.api_type,
                model_name=config.model_name,
                verbose=verbose,
                base_url=config.base_url,
                api_key_env=config.api_key_env,
            )
            clients.append(client)
        return clients


__all__ = [
    "LLMClient",
    "LLMRetryAudit",
    "get_retry_audit",
    "clear_retry_audit",
    "NetworkRecoveryConfig",
    "get_network_config",
    "set_network_config",
]
