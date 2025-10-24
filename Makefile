
PY=.venv/bin/python
PIP=.venv/bin/pip

.PHONY: setup mlflow-ui train predict submit clean


setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

train:
	@echo "[make] TRAIN"
	@mkdir -p outputs/models outputs/logs outputs/submissions
	$(PY) -m src.train --config configs/base.yaml

predict:
	@echo "[make] PREDICT"
	@mkdir -p outputs/submissions
	$(PY) -m src.predict --config configs/base.yaml

submit:
	@ls -1 outputs/submissions/*.csv | tail -n1

clean:
	@echo "[make] CLEAN"
	rm -rf outputs/models outputs/logs outputs/submissions
