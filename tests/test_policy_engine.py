from config.schema import ProjectConfig
from core.policy_engine import decide
from core.types import HarmVerdict, InjectionVerdict, PIIVerdict, ScopeVerdict


def _config(**overrides) -> ProjectConfig:
    defaults = {"project_id": "p1", "scope_definition": "test", "harm_severity_block_threshold": 4}
    defaults.update(overrides)
    return ProjectConfig(**defaults)


def _clean_verdicts():
    return dict(
        injection=InjectionVerdict(flagged=False, confidence=0.1),
        pii=PIIVerdict(entities=[], masked_text="hello", masked_entity_ids=[]),
        scope=ScopeVerdict(in_scope=True, confidence=0.9),
        harm=HarmVerdict(category_scores={}, severity=0),
    )


def test_allow_when_everything_clean():
    decision = decide(**_clean_verdicts(), project_config=_config())
    assert decision.action == "allow"


def test_injection_takes_precedence_over_everything():
    verdicts = _clean_verdicts()
    verdicts["injection"] = InjectionVerdict(flagged=True, confidence=0.99, reason="jailbreak attempt")
    verdicts["harm"] = HarmVerdict(category_scores={"hate": 0.9}, severity=6)  # would also block
    verdicts["scope"] = ScopeVerdict(in_scope=False, confidence=0.9)  # would also block

    decision = decide(**verdicts, project_config=_config())
    assert decision.action == "block"
    assert "injection" in decision.reasons[0]


def test_harm_blocks_when_severity_at_or_above_threshold():
    verdicts = _clean_verdicts()
    verdicts["harm"] = HarmVerdict(category_scores={"hate": 0.9}, severity=4)

    decision = decide(**verdicts, project_config=_config(harm_severity_block_threshold=4))
    assert decision.action == "block"
    assert "harm" in decision.reasons[0]


def test_harm_below_threshold_does_not_block():
    verdicts = _clean_verdicts()
    verdicts["harm"] = HarmVerdict(category_scores={"hate": 0.5}, severity=2)

    decision = decide(**verdicts, project_config=_config(harm_severity_block_threshold=4))
    assert decision.action == "allow"


def test_scope_blocks_when_out_of_scope_and_harm_clean():
    verdicts = _clean_verdicts()
    verdicts["scope"] = ScopeVerdict(in_scope=False, confidence=0.8, reason="no close in-scope match")

    decision = decide(**verdicts, project_config=_config())
    assert decision.action == "block"
    assert "scope" in decision.reasons[0]


def test_pii_masks_when_in_scope_and_not_harmful():
    verdicts = _clean_verdicts()
    verdicts["pii"] = PIIVerdict(entities=[], masked_text="my [PHONE] is masked", masked_entity_ids=["PHONE_NUMBER:0:5"])

    decision = decide(**verdicts, project_config=_config())
    assert decision.action == "mask"
    assert decision.masked_text == "my [PHONE] is masked"


def test_scope_takes_precedence_over_pii_masking():
    verdicts = _clean_verdicts()
    verdicts["scope"] = ScopeVerdict(in_scope=False, confidence=0.8)
    verdicts["pii"] = PIIVerdict(entities=[], masked_text="masked", masked_entity_ids=["X:0:1"])

    decision = decide(**verdicts, project_config=_config())
    assert decision.action == "block"
