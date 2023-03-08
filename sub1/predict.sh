#!/bin/bash

python3 setfinal.py
python3 predict.py

cp *onnx /output/

