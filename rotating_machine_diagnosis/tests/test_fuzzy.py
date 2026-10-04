"""Tests for fuzzy logic risk assessment system."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FuzzyConfig
from fuzzy.fuzzy_system import FuzzyRiskAssessor
from fuzzy.rules import get_default_rules, rules_to_text


@pytest.fixture
def assessor():
    return FuzzyRiskAssessor()


class TestFuzzySystem:
    def test_low_risk(self, assessor):
        """Low fault prob + low RMS + low kurtosis → low risk."""
        result = assessor.assess_risk(
            fault_probability=0.1,
            rms=0.1,
            kurtosis=2.0,
        )
        assert result["risk_score"] < 40
        assert result["severity"] in ("LOW", "MEDIUM")

    def test_high_risk(self, assessor):
        """High fault prob + high RMS + high kurtosis → critical risk."""
        result = assessor.assess_risk(
            fault_probability=0.95,
            rms=0.9,
            kurtosis=15.0,
        )
        assert result["risk_score"] > 60
        assert result["severity"] in ("HIGH", "CRITICAL")

    def test_risk_score_range(self, assessor):
        """Risk score should always be in [0, 100]."""
        for fp in [0.01, 0.5, 0.99]:
            for rms in [0.01, 0.5, 0.99]:
                for kurt in [0.5, 5.0, 15.0]:
                    result = assessor.assess_risk(fp, rms, kurt)
                    assert 0 <= result["risk_score"] <= 100

    def test_result_keys(self, assessor):
        result = assessor.assess_risk(0.5, 0.5, 5.0)
        assert "risk_score" in result
        assert "severity" in result
        assert "recommendation" in result
        assert "membership_values" in result
        assert "inputs" in result

    def test_severity_levels(self, assessor):
        """Severity should be one of the defined levels."""
        valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        for _ in range(20):
            fp = np.random.uniform(0.01, 0.99)
            rms = np.random.uniform(0.01, 0.99)
            kurt = np.random.uniform(0.5, 18.0)
            result = assessor.assess_risk(fp, rms, kurt)
            assert result["severity"] in valid

    def test_recommendation_not_empty(self, assessor):
        result = assessor.assess_risk(0.5, 0.5, 5.0)
        assert len(result["recommendation"]) > 0

    def test_membership_values_structure(self, assessor):
        result = assessor.assess_risk(0.5, 0.5, 5.0)
        memb = result["membership_values"]
        assert "rms" in memb
        assert "kurtosis" in memb
        assert "fault_probability" in memb
        # Each should have low/medium/high terms
        for var_name, terms in memb.items():
            assert "low" in terms
            assert "medium" in terms
            assert "high" in terms
            # Values should be in [0, 1]
            for term, val in terms.items():
                assert 0.0 <= val <= 1.0 + 1e-10, \
                    f"{var_name}.{term} = {val} out of [0, 1]"

    def test_monotonic_risk(self, assessor):
        """Higher fault probability should generally produce higher risk."""
        r_low = assessor.assess_risk(0.1, 0.5, 5.0)
        r_high = assessor.assess_risk(0.9, 0.5, 5.0)
        assert r_high["risk_score"] >= r_low["risk_score"]

    def test_custom_config(self):
        config = FuzzyConfig()
        assessor = FuzzyRiskAssessor(config=config)
        result = assessor.assess_risk(0.5, 0.5, 5.0)
        assert isinstance(result["risk_score"], float)


class TestFuzzyRules:
    def test_get_default_rules(self):
        rules = get_default_rules()
        assert len(rules) > 0
        for rule in rules:
            assert "antecedents" in rule
            assert "consequent" in rule

    def test_rules_to_text(self):
        rules = get_default_rules()
        text = rules_to_text(rules)
        assert "Rule 1" in text
        assert "IF" in text
        assert "THEN" in text

    def test_rules_are_copies(self):
        """get_default_rules should return a copy, not a reference."""
        r1 = get_default_rules()
        r2 = get_default_rules()
        r1[0]["weight"] = 999
        assert r2[0]["weight"] != 999

    def test_all_rules_have_weight(self):
        rules = get_default_rules()
        for rule in rules:
            assert "weight" in rule
            assert 0 < rule["weight"] <= 1.0
