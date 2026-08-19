.PHONY: smoke test train serve clean

# End-to-end offline smoke: train + exercise the predict/serve path.
smoke:
	python scripts/smoke.py

# Unit tests.
test:
	pytest -q

# Train the model from configs/default.yaml (writes artifacts/model.pkl).
train:
	python -m src.train --config configs/default.yaml

# Run the Flask web app locally (optional, needs a browser).
serve:
	python app.py

clean:
	rm -rf artifacts __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
