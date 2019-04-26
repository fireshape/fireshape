.PHONY: test examples lint

test:
	pytest

examples:
	cd examples/L2tracking/; python3 L2tracking_main.py
	cd examples/levelset/; python3 levelset.py; python3 levelset_boundary.py; python3 levelset_multigrid.py; python3 levelset_spline.py

lint:
	flake8
