#!/bin/bash

cd src
python downloader.py --download cnn_features --dataset-name flickr8k --cnn-name googlenet
python downloader.py --download captions --dataset-name flickr8k
python downloader.py --download images --dataset-name flickr8k
cd ..
