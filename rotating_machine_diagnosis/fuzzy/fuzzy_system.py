"""
Fuzzy Inference System — Mamdani fuzzy logic for machine risk assessment.

Uses scikit-fuzzy for membership functions and defuzzification.
The system takes deep-learning fault probability + traditional vibration
features and produces a risk score (0–100) with severity level.

Architecture:
    DL Fault Probability  ─┐
    RMS                    ─┤→ Fuzzy Inference → Risk Score (0–100)
    Kurtosis               ─┘      │
                                   ↓
                         Low / Medium / High / Critical
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Dict, Optional, Tuple, List, Any

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FuzzyConfig, RECOMMENDATION_THRESHOLDS, RECOMMENDATION_MESSAGES
from fuzzy.rules import get_default_rules


class FuzzyRiskAssessor:
    """
    Mamdani fuzzy inference system for machine risk assessment.

    Inputs:
        - fault_probability (0–1): from DL classifier
        - rms (0–1): normalized RMS
        - kurtosis (0–20): kurtosis value

    Output:
        - risk (0–100): machine risk score
    """

    def __init__(self, config: Optional[FuzzyConfig] = None, rules: Optional[List[Dict]] = None):
        """
        Initialize the fuzzy system with configurable membership functions and rules.

        Parameters
        ----------
        config : FuzzyConfig or None
            Membership function parameters. Uses defaults if None.
        rules : list of dict or None
            Custom rules. Uses default rules if None.
        """
        self.config = config or FuzzyConfig()
        self.rules_data = rules or get_default_rules()
        self._build_system()

    def _build_system(self):
        """Build the fuzzy control system."""
        cfg = self.config

        # ── Define universes ──
        rms_range = np.linspace(cfg.rms_universe[0], cfg.rms_universe[1], 1000)
        kurtosis_range = np.linspace(cfg.kurtosis_universe[0], cfg.kurtosis_universe[1], 1000)
        fp_range = np.linspace(cfg.fault_prob_universe[0], cfg.fault_prob_universe[1], 1000)
        risk_range = np.linspace(cfg.risk_universe[0], cfg.risk_universe[1], 1000)

        # ── Create Antecedent/Consequent objects ──
        self.rms_var = ctrl.Antecedent(rms_range, "rms")
        self.kurtosis_var = ctrl.Antecedent(kurtosis_range, "kurtosis")
        self.fp_var = ctrl.Antecedent(fp_range, "fault_probability")
        self.risk_var = ctrl.Consequent(risk_range, "risk")

        # ── Membership functions: RMS ──
        self.rms_var["low"] = fuzz.trimf(rms_range, cfg.rms_low)
        self.rms_var["medium"] = fuzz.trimf(rms_range, cfg.rms_medium)
        self.rms_var["high"] = fuzz.trimf(rms_range, cfg.rms_high)

        # ── Membership functions: Kurtosis ──
        self.kurtosis_var["low"] = fuzz.trimf(kurtosis_range, cfg.kurtosis_low)
        self.kurtosis_var["medium"] = fuzz.trimf(kurtosis_range, cfg.kurtosis_medium)
        self.kurtosis_var["high"] = fuzz.trimf(kurtosis_range, cfg.kurtosis_high)

        # ── Membership functions: Fault Probability ──
        self.fp_var["low"] = fuzz.trimf(fp_range, cfg.fault_prob_low)
        self.fp_var["medium"] = fuzz.trimf(fp_range, cfg.fault_prob_medium)
        self.fp_var["high"] = fuzz.trimf(fp_range, cfg.fault_prob_high)

        # ── Membership functions: Risk (output) ──
        self.risk_var["low"] = fuzz.trapmf(risk_range, cfg.risk_low)
        self.risk_var["medium"] = fuzz.trapmf(risk_range, cfg.risk_medium)
        self.risk_var["high"] = fuzz.trapmf(risk_range, cfg.risk_high)
        self.risk_var["critical"] = fuzz.trapmf(risk_range, cfg.risk_critical)

        # Defuzzification method
        self.risk_var.defuzzify_method = "centroid"

        # ── Build rules ──
        var_map = {
            "rms": self.rms_var,
            "kurtosis": self.kurtosis_var,
            "fault_probability": self.fp_var,
            "risk": self.risk_var,
        }

        fuzzy_rules = []
        for rule_data in self.rules_data:
            antecedents = rule_data["antecedents"]
            con_var, con_term = rule_data["consequent"]

            # Build antecedent expression
            ant_expr = None
            for var_name, term_name in antecedents:
                if var_name not in var_map:
                    continue
                term_expr = var_map[var_name][term_name]
                if ant_expr is None:
                    ant_expr = term_expr
                else:
                    ant_expr = ant_expr & term_expr

            if ant_expr is not None and con_var in var_map:
                consequent = var_map[con_var][con_term]
                fuzzy_rules.append(ctrl.Rule(ant_expr, consequent))

        if not fuzzy_rules:
            raise ValueError("No valid fuzzy rules could be constructed.")

        self.control_system = ctrl.ControlSystem(fuzzy_rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    def assess_risk(
        self,
        fault_probability: float,
        rms: float,
        kurtosis: float,
    ) -> Dict[str, Any]:
        """
        Perform fuzzy risk assessment.

        Parameters
        ----------
        fault_probability : float
            Highest class probability from the DL model (0–1).
        rms : float
            Normalized RMS value (0–1).
        kurtosis : float
            Kurtosis value (clipped to universe range).

        Returns
        -------
        dict with keys:
            risk_score, severity, recommendation,
            membership_values, inputs
        """
        cfg = self.config

        # Clip inputs to universe ranges
        fp_clipped = float(np.clip(fault_probability, cfg.fault_prob_universe[0] + 0.001,
                                   cfg.fault_prob_universe[1] - 0.001))
        rms_clipped = float(np.clip(rms, cfg.rms_universe[0] + 0.001,
                                    cfg.rms_universe[1] - 0.001))
        kurt_clipped = float(np.clip(kurtosis, cfg.kurtosis_universe[0] + 0.001,
                                     cfg.kurtosis_universe[1] - 0.001))

        # Set inputs
        self.simulation.input["fault_probability"] = fp_clipped
        self.simulation.input["rms"] = rms_clipped
        self.simulation.input["kurtosis"] = kurt_clipped

        # Compute
        try:
            self.simulation.compute()
            risk_score = float(self.simulation.output["risk"])
        except Exception:
            # Fallback: simple weighted average
            risk_score = float(
                fp_clipped * 40 + rms_clipped * 40 + (kurt_clipped / 20.0) * 20
            )

        risk_score = float(np.clip(risk_score, 0, 100))

        # Determine severity
        severity = self._score_to_severity(risk_score)

        # Get recommendation
        recommendation = self._get_recommendation(risk_score)

        # Compute membership values for explainability
        membership_values = self._compute_memberships(fp_clipped, rms_clipped, kurt_clipped)

        return {
            "risk_score": round(risk_score, 1),
            "severity": severity,
            "recommendation": recommendation,
            "membership_values": membership_values,
            "inputs": {
                "fault_probability": fp_clipped,
                "rms": rms_clipped,
                "kurtosis": kurt_clipped,
            },
        }

    def _score_to_severity(self, score: float) -> str:
        """Map a 0–100 risk score to a severity label."""
        if score < 25:
            return "LOW"
        elif score < 50:
            return "MEDIUM"
        elif score < 75:
            return "HIGH"
        else:
            return "CRITICAL"

    def _get_recommendation(self, score: float) -> str:
        """Get maintenance recommendation based on risk score."""
        for level, (lo, hi) in RECOMMENDATION_THRESHOLDS.items():
            if lo <= score < hi:
                return RECOMMENDATION_MESSAGES[level]
        return RECOMMENDATION_MESSAGES["immediate"]

    def _compute_memberships(
        self, fp: float, rms: float, kurtosis: float
    ) -> Dict[str, Dict[str, float]]:
        """Compute membership values for all input variables."""
        cfg = self.config
        memberships = {}

        # RMS
        rms_range = np.array([rms])
        memberships["rms"] = {
            "low": float(fuzz.trimf(rms_range, cfg.rms_low)[0]),
            "medium": float(fuzz.trimf(rms_range, cfg.rms_medium)[0]),
            "high": float(fuzz.trimf(rms_range, cfg.rms_high)[0]),
        }

        # Kurtosis
        kurt_range = np.array([kurtosis])
        memberships["kurtosis"] = {
            "low": float(fuzz.trimf(kurt_range, cfg.kurtosis_low)[0]),
            "medium": float(fuzz.trimf(kurt_range, cfg.kurtosis_medium)[0]),
            "high": float(fuzz.trimf(kurt_range, cfg.kurtosis_high)[0]),
        }

        # Fault probability
        fp_range = np.array([fp])
        memberships["fault_probability"] = {
            "low": float(fuzz.trimf(fp_range, cfg.fault_prob_low)[0]),
            "medium": float(fuzz.trimf(fp_range, cfg.fault_prob_medium)[0]),
            "high": float(fuzz.trimf(fp_range, cfg.fault_prob_high)[0]),
        }

        return memberships
