get-data:
	kaggle competitions download -c amex-default-prediction -p ./data/
	unzip ./data/amex-default-prediction.zip -d ./data/
.PHONY: get-data