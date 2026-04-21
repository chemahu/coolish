.PHONY: all data clean 01 02 03 04 05 06 07 08 09

DATA_TAG ?= latest
BARS ?=

all: data 01 02 03 04 05 06 07 08

data:
	bash scripts/download_data.sh --tag $(DATA_TAG)

01:
	python scripts/01_load_and_index.py

02:
	python scripts/02_yearly_summary.py

03:
	python scripts/03_fifo_roundtrip.py

04:
	python scripts/04_pyramid_fingerprint.py

05:
	python scripts/05_leverage_drawdown.py

06:
	python scripts/06_funding_cashflow.py

07:
	python scripts/07_counterfactual.py

08:
	python scripts/08_strategy_spec.py

09:
	python scripts/09_backtest_replica.py $(if $(BARS),--bars $(BARS),--synthetic)

clean:
	rm -rf outputs/* data/*
	rm -rf outputs/cache/
