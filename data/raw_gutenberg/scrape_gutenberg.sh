#!/bin/bash

for ((id=68460; id<=71460; id++)); do
    python scrape_gutenberg.py --data-directory='./XX/' --action=$id
done
