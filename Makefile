PYBIND_DIR := $(shell python3.11 -c "import pybind11; print(pybind11.get_cmake_dir())")

build:
	mkdir -p build
	cd build && cmake -Dpybind11_DIR=$(PYBIND_DIR) .. && make -j
	cd python && python test_pybind.py

rebuild:
	rm -rf build/*
	cd build && cmake -Dpybind11_DIR=$(PYBIND_DIR) .. && make -j
	cd python && python test_pybind.py

clean:
	rm -rf build/*

.PHONY: build rebuild clean