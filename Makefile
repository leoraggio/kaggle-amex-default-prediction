fetch-data:
	@echo "Fetching data from Kaggle.com..."
	kaggle competitions download -c ${COMPETITION} -p ./data/
.PHONY: get-data

unzip-data:
	@echo "Extracting data..."
	unzip ./data/amex-default-prediction.zip -d ./data/
.PHONY: unzip-data

features:
	python -m research.features

train:
	python -m research.train

update-env:
	umask 0002 && /opt/conda/bin/mamba env update -n amex -f environment.yml
