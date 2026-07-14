#!/bin/bash


python run.py --name gemma-4-31b-d1 --model gemma-4-31b --diff 1 --natoms 20 --volume 8000 --temp 2000 --max-steps 5000 --seed 1337 
python run.py --name gemma-4-26b-a3b-d1 --model gemma-4-26b-a4b --diff 1 --natoms 20 --volume 8000 --temp 2000 --max-steps 5000 --seed 1337 
python run.py --name gpt-oss-20b-d1 --model gpt-oss-20b --diff 1 --natoms 20 --volume 8000 --temp 2000 --max-steps 5000 --seed 1337 
python run.py --name gpt-oss-120b-d1 --model gpt-oss-120b --diff 1 --natoms 20 --volume 8000 --temp 2000 --max-steps 5000 --seed 1337 
python run.py --name qwen36-27b-d1 --model qwen3.6-27b --diff 1 --natoms 20 --volume 8000 --temp 2000 --max-steps 5000 --seed 1337 
python run.py --name qwen35-122b-a10b-d1 --model qwen3.5-122b-a10b --diff 1 --natoms 20 --volume 8000 --temp 2000 --max-steps 5000 --seed 1337 
python run.py --name mistral-medium-35-128b-d1 --model mistral-medium-3.5-128b --diff 1 --natoms 20 --volume 8000 --temp 2000 --max-steps 5000 --seed 1337 

