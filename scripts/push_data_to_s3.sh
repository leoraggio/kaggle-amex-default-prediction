#!/bin/sh

BUCKET="s3://cora-data-test-custom"
BUCKET_PREFIX="user_leoraggio"
COMPETITION="amex-default-prediction"
S3_FULL_PATH="${BUCKET}/${BUCKET_PREFIX}/${COMPETITION}"

for file in data/*.csv; do \
  FNAME=$(basename ${file})
  echo pushing ${FNAME} to ${S3_FULL_PATH}; \
  aws s3 cp ${file} ${S3_FULL_PATH}/${FNAME}
done
