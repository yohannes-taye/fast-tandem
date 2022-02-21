rm -r /home/tmc/output_tandem_replica/*
export TANDEM_DATA_DIR=/home/tmc/tandem_replica
python train.py --config /home/tmc/tandem/cva_mvsnet/configs/default.yaml /home/tmc/output_tandem_replica DATA.ROOT_DIR $TANDEM_DATA_DIR