
fetch-data:
	@echo "Fetching data from Kaggle.com..."
	kaggle competitions download -c ${COMPETITION} -p ./data/
.PHONY: get-data

unzip-data:
	@echo "Extracting data..."
	unzip ./data/amex-default-prediction.zip -d ./data/
.PHONY: unzip-data

push-data-to-s3:
	@./scripts/push_data_to_s3.sh
.PHONY: push-data-to-s3

features:
	python -m research.features

update-env:
	umask 0002 && /opt/conda/bin/mamba env update -n amex -f environment.yml
