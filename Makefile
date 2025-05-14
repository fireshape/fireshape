.PHONY: test examples lint

test:
	pytest

examples:
	cd examples/L2tracking/; python3 L2tracking_main.py
	cd examples/levelset/; python3 levelset.py; python3 levelset_boundary.py; python3 levelset_multigrid.py;
	cd examples/levelset/; python3 levelset_spline.py; python3 levelset_fedecoupled.py


lint:
	flake8 --ignore=F403,F405,E226
