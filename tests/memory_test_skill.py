from __future__ import annotations

import re

BLOCKED_SKILL_PATTERNS = [
    # Shell execution / process spawning
    (re.compile(r"\b(child_process|subprocess|os\.system|os\.exec|os\.popen|exec\(|eval\()\b"), "code execution"),
    # Crypto mining
    (re.compile(r"\b(stratum|mining|hashrate|cryptonight|xmrig)\b", re.IGNORECASE), "crypto mining"),
    # Credential harvesting
    (
        re.compile(
            r"\b(password|credential|keylog|clipboard\.read)\b.*\b(send|post|upload|exfiltrate)\b", re.IGNORECASE
        ),
        "credential harvesting",
    ),
    # Network exfiltration
    (
        re.compile(
            r"\b(curl|wget|fetch|requests\.post|httpx\.post)\b.*\b(api_key|secret|token|password|credential)\b",
            re.IGNORECASE,
        ),
        "data exfiltration",
    ),
    # Reverse shell patterns
    (re.compile(r"/dev/tcp/|/dev/udp/|nc\s+-[el]|netcat|socat|telnet\s+\d"), "reverse shell"),
    # Base64-encoded payloads (obfuscation)
    (re.compile(r"base64[._-]?(?:decode|b64decode)\s*\("), "obfuscated code"),
    # Self-modification of system prompt / core identity
    (
        re.compile(
            r"\b(system_prompt|CORE_IDENTITY|BASE_SYSTEM_PROMPT)\b.*\b(replace|overwrite|delete|modify)\b",
            re.IGNORECASE,
        ),
        "system prompt modification",
    ),
]

def validate_skill_content(instructions: str) -> list[str]:
    """Scan skill instructions for dangerous patterns.

    Returns list of violation descriptions. Empty list = safe.
    OpenClaw-inspired: prevents persistent prompt injection via skills.
    Normalizes homoglyphs first to prevent Cyrillic bypass.
    Also checks for prompt injection patterns (e.g. "ignore all previous instructions").
    """
    from memory_test_sanitizer import detect_prompt_injection, normalize_homoglyphs

    normalized = normalize_homoglyphs(instructions)
    violations = []
    for pattern, description in BLOCKED_SKILL_PATTERNS:
        if pattern.search(normalized):
            violations.append(description)

    # Also check for prompt injection patterns
    injection_warnings = detect_prompt_injection(normalized)
    if injection_warnings:
        violations.extend([f"prompt injection: {w}" for w in injection_warnings])

    return violations
