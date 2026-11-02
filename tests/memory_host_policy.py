"""Explicit pass-through policy for synthetic test inputs only."""


def synthetic_policy(**overrides):
    return {
        "scrub_secrets": lambda text: text,
        "validate_skill_content": lambda text: [],
        "redact_urls_for_trace": lambda text: text,
        **overrides,
    }
