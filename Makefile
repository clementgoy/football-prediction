
PY=.venv/bin/python
PIP=.venv/bin/pip
TRAIN_DIR ?= data/Train_Data
TEST_DIR  ?= data/Test_Data
Y_TRAIN   ?= data/Y_train_1rknArQ.csv
OUT_DIR   ?= data/processed

.PHONY: setup mlflow-ui train predict submit clean

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

.PHONY: merge
merge:
	@echo "Merging raw CSVs into modelling tables"
	$(PY) -m src.merge_data \
		--train-dir $(TRAIN_DIR) \
		--test-dir  $(TEST_DIR) \
		--y-train   $(Y_TRAIN) \
		--out-dir   $(OUT_DIR)
	@echo "Done. Files in $(OUT_DIR)/ (train_merged.csv, test_merged.csv, schema.json)"

train-hgbc:
	@echo "[make] TRAIN"
	@mkdir -p outputs/models outputs/logs outputs/submissions
	$(PY) -m src.train_hgbc --config configs/base.yaml

train-lgbm:
	@echo "[make] TRAIN"
	$(PY) -m src.train_lgbm \
		--train-csv data/processed/train_merged.csv \
		--y-csv    data/processed/y_train_aligned.csv \
		--model-out outputs/models/lgbm.pkl

predict:
	@echo "[make] PREDICT"
	$(PY) -m src.predict \
		--test-csv data/processed/test_merged.csv \
		--artifact outputs/models/model.joblib \
		--out-csv outputs/submissions/submission_hgbc.csv \
		--class-order HOME_WINS,DRAW,AWAY_WINS


submit:
	@ls -1 outputs/submissions/*.csv | tail -n1

clean:
	@echo "[make] CLEAN"
	rm -rf outputs/models outputs/logs outputs/submissions
