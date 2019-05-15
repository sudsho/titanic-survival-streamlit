"""Model factory."""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def build_model(cfg):
    m = cfg.get("model", {})
    name = m.get("type", "random_forest")
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=m.get("n_estimators", 100),
            max_depth=m.get("max_depth", 6),
            min_samples_split=m.get("min_samples_split", 2),
            min_samples_leaf=m.get("min_samples_leaf", 1),
            random_state=42,
            n_jobs=-1,
        )
    if name == "gbm":
        return GradientBoostingClassifier(
            n_estimators=m.get("n_estimators", 100),
            max_depth=m.get("max_depth", 3),
            learning_rate=m.get("learning_rate", 0.1),
            random_state=42,
        )
    if name == "logreg":
        return LogisticRegression(
            C=m.get("C", 1.0),
            solver="liblinear",
            random_state=42,
        )
    raise ValueError("unknown model: " + name)
