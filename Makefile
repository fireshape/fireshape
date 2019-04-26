.PHONY: test, examples, lint

test:
	pytest test/

examples:
	cd examples/L2tracking/; python3 L2tracking_main.py; cd -
	cd examples/levelset/; python3 levelset.py; cd -

lint:
	echo "Not setup yet"


