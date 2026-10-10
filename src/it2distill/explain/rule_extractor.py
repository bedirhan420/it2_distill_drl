"""Turns a trained FuzzyPolicyNet into readable IF-THEN rules."""
from __future__ import annotations

from dataclasses import dataclass

from it2distill.fuzzy.model import FuzzyPolicyNet


@dataclass
class ExtractedRule:
    antecedent: list[str]        # ["cart_pos is low (FOU 0.2)", ...]
    consequent: str              # "push right" or "a = 0.3*x1 - 0.1*x2 + 0.05"
    firing_note: str = ""
    support: float = 0.0         # mean normalised firing on train data (rule importance)

    def __str__(self) -> str:
        return "IF " + " AND ".join(self.antecedent) + f" THEN {self.consequent}"


class RuleExtractor:
    def extract(self, model: FuzzyPolicyNet, labeler=None, x_train=None) -> list[ExtractedRule]:
        """TODO(WP6): use model.fuzzify.mf.center_and_width() + labeler to name terms; drop terms whose
        sigma spans the whole feature range ('don't care'); rank rules by support."""
        raise NotImplementedError
