"""Model factory."""
from sklearn.ensemble import RandomForestClassifier


def build_model(cfg):
    m = cfg.get("model", {})
    return RandomForestClassifier(
        n_estimators=m.get("n_estimators", 100),
        max_depth=m.get("max_depth", 6),
        min_samples_split=m.get("min_samples_split", 2),
        random_state=42,
    )
