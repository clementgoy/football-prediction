PY=.venv/bin/python
PIP=.venv/bin/pip
TRAIN_DIR ?= data/Train_Data
TEST_DIR  ?= data/Test_Data
Y_TRAIN   ?= data/Y_train_1rknArQ.csv
OUT_DIR   ?= data/processed

.PHONY: setup mlflow-ui merge train-hgbc train-lgbm predict-hgbc predict-lgbm submit clean

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

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
	@mkdir -p models
	$(PY) -m src.train_hgbc --config configs/base.yaml

train-lgbm:
	@echo "[make] TRAIN"
	$(PY) -m src.train_lgbm \
		--train-csv data/processed/train_merged.csv \
		--y-csv    data/processed/y_train_aligned.csv \
		--model-out models/lgbm.pkl

train-goal-diff-lgbm:
	@echo "[make] TRAIN"
	$(PY) -m src.train_goal_diff_lgbm \
	--train-csv data/processed/train_merged.csv \
	--y-supp-csv data/processed/y_train_supp_aligned.csv \
	--model-out models/lgbm_goal_diff.pkl

predict-hgbc:
	@echo "[make] PREDICT"
	$(PY) -m src.predict \
		--test-csv data/processed/test_merged.csv \
		--artifact outputs/models/model.joblib \
		--out-csv outputs/submissions/submission_hgbc.csv \
		--class-order HOME_WINS,DRAW,AWAY_WINS

predict-lgbm:
	@echo "[make] PREDICT"
	$(PY) -m src.predict_lgbm \
		--test-csv data/processed/test_merged.csv \
		--model outputs/models/lgbm.pkl \
		--out-csv outputs/submissions/submission_lgbm.csv \
		--id-col ID \
		--alpha-draw 1.0 \
		--submit-onehot

predict-goal-diff-lgbm:
	@echo "[make] PREDICT"
	$(PY) -m src.predict_goal_diff_lgbm \
		--test-csv data/processed/test_merged.csv \
		--model models/lgbm_goal_diff.pkl \
		--out-csv models/submission_lgbm_goal_diff.csv

submit:
	@ls -1 outputs/submissions/*.csv | tail -n1

clean:
	@echo "[make] CLEAN"
	rm -rf models/*
