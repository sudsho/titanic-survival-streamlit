"""Train the titanic model."""
import argparse
import logging
import joblib
from sklearn.model_selection import train_test_split
from src.preprocess import load_csv, make_features
from src.model import build_model
from src.utils import load_config, ensure_dir
from src.eval import report


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(cfg_path):
    cfg = load_config(cfg_path)
    df = load_csv(cfg["data"]["train_csv"])
    df = make_features(df)
    y = df["Survived"].values
    feat_cols = [c for c in df.columns if c != "Survived"]
    X = df[feat_cols].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )
    model = build_model(cfg)
    model.fit(X_tr, y_tr)
    log.info("train acc: %.3f", model.score(X_tr, y_tr))
    log.info("test acc: %.3f", model.score(X_te, y_te))

    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    metrics = report(y_te, y_pred, y_proba)
    for k, v in metrics.items():
        log.info("%s: %.3f", k, v)

    out = cfg["artifacts"]["model_path"]
    ensure_dir(out)
    joblib.dump({"model": model, "feat_cols": feat_cols}, out)
    log.info("saved model -> %s", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)
