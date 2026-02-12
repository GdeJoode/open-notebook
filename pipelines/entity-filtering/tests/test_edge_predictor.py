"""Tests for EdgePredictor stub implementation."""

from entity_filtering.scoring.edge_predictor import EdgePredictor


def _entity(text, label="MISC", confidence=0.9):
    """Helper to build an entity dict."""
    return {
        "text": text,
        "label": label,
        "properties": {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


def _relation(source, target, rel_type="RELATED_TO", confidence=0.8):
    """Helper to build a relation dict."""
    return {
        "source_entity": source,
        "target_entity": target,
        "relation_type": rel_type,
        "properties": {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


class TestEdgePredictorStub:
    def test_predict_returns_empty_list(self):
        predictor = EdgePredictor()
        entities = [_entity("Alice"), _entity("Bob")]
        relations = [_relation("Alice", "Bob", "KNOWS")]
        result = predictor.predict(entities, relations)
        assert result == []

    def test_predict_with_empty_inputs(self):
        predictor = EdgePredictor()
        result = predictor.predict([], [])
        assert result == []

    def test_predict_with_config(self):
        predictor = EdgePredictor(config={"model": "test"})
        entities = [_entity("X")]
        result = predictor.predict(entities, [])
        assert result == []

    def test_default_config_is_empty_dict(self):
        predictor = EdgePredictor()
        assert predictor._config == {}

    def test_custom_config_stored(self):
        config = {"threshold": 0.5, "max_predictions": 10}
        predictor = EdgePredictor(config=config)
        assert predictor._config == config
