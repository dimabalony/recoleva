"""Cross-script (Cyrillic↔Latin) entity-name canonicalization.

Sub-design C deferred-cluster (design §5): a PRECISION-clean transliteration
key so cross-script variants of the same name collapse to one entity, WITHOUT
ever false-merging two genuinely distinct people (the catastrophic error). When
uncertain, the function UNDER-merges.

Scope boundary (deliberate): clean transliteration only. Diminutive↔formal
(``Миша``↔``Михаил``) and spelling/ending variants (``Патров``↔``Petrov``,
``-ау``↔``-ов``) are NOT handled here — those are nickname/fuzzy territory and
carry false-merge risk, so they stay out of scope.
"""

from __future__ import annotations

from kai_memory.canonicalization import (
    same_canonical_entity,
    translit_key,
)


class TestTranslitKeyCollapsesCleanTransliterations:
    def test_basic_cyrillic_latin_pair_shares_key(self):
        assert translit_key("Миша") == translit_key("Misha")

    def test_soft_sign_apostrophe_is_dropped(self):
        # unidecode("Пермь") == "Perm'" — the soft-sign apostrophe must not
        # split it from the user-typed "Perm".
        assert translit_key("Пермь") == translit_key("Perm")

    def test_iotated_vowel_scheme_is_folded(self):
        # unidecode("Катя") == "Nastia"; users type "Katya" (я→ya vs ia).
        assert translit_key("Катя") == translit_key("Katya")

    def test_final_y_folds_to_i(self):
        # unidecode("Нора") == "Betti"; user "Nora".
        assert translit_key("Нора") == translit_key("Nora")

    def test_case_insensitive(self):
        assert translit_key("МИША") == translit_key("misha")

    def test_already_ascii_passthrough_is_stable(self):
        assert translit_key("Minsk") == translit_key("Минск")


class TestTranslitKeyKeepsDistinctNamesApart:
    def test_distinct_people_have_distinct_keys(self):
        assert translit_key("Миша") != translit_key("Ольга")

    def test_diminutive_vs_formal_stay_distinct(self):
        # Out-of-scope by design: must NOT collapse (nickname, not translit).
        assert translit_key("Миша") != translit_key("Михаил")

    def test_spelling_variant_stays_distinct(self):
        # Belarusian а vs canonical о — out of scope, must not collapse.
        assert translit_key("Патров") != translit_key("Petrov")

    def test_near_miss_hardened_negatives_stay_distinct(self):
        for a, b in [
            ("Анна", "Аня"),
            ("Олег", "Олеся"),
            ("Алёна", "Елена"),
            ("Pavel", "Павло"),
            ("Катя", "Наташа"),
        ]:
            assert translit_key(a) != translit_key(b), f"{a!r}/{b!r} must not collapse"


class TestTranslitKeyEmptyAndGarbage:
    def test_empty_string_is_empty_key(self):
        assert translit_key("") == ""

    def test_punctuation_only_is_empty_key(self):
        assert translit_key("!!!") == ""


class TestSameCanonicalEntity:
    def test_same_name_same_type_merges(self):
        assert same_canonical_entity("Миша", "Misha", "person", "person") is True

    def test_cross_script_without_types_merges(self):
        assert same_canonical_entity("Катя", "Katya") is True

    def test_type_mismatch_blocks_merge(self):
        # agent-Kai must NEVER merge into person-Kai (entity-typing cluster).
        assert same_canonical_entity("Kai", "Kai", "agent", "person") is False

    def test_distinct_people_never_merge(self):
        assert same_canonical_entity("Миша", "Ольга", "person", "person") is False

    def test_empty_key_never_merges(self):
        # Two surfaces that both reduce to an empty key must not be "equal".
        assert same_canonical_entity("!!!", "???") is False
