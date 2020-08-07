#!/bin/bash

export python_path=$(which python)

echo $python_path

sudo $python_path -m nltk.downloader -d /usr/local/share/nltk_data all


