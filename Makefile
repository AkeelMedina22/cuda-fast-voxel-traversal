build:
	mkdir -p build
	cd build && cmake .. && make -j
	cd python && python main.py

rebuild:
	rm -rf build/*
	cd build && cmake .. && make -j
	cd python && python main.py

clean:
	rm -rf build/*

.PHONY: build rebuild clean