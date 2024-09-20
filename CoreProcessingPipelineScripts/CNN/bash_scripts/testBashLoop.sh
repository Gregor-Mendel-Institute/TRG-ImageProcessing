#!/usr/local/bin/env bash
#bash --version
DOWNSAMPLED_FOLDER='.'
CORES=("measureCoresCrac" "retraining")

TIFS=()
for c in ${CORES[@]}; do
  echo $c
  #TIFS+="${c} "
  TIFS+="$(find ${DOWNSAMPLED_FOLDER} -name $c"*.sh" ) "
done
echo $TIFS

for i in $TIFS; do
  echo $i
done