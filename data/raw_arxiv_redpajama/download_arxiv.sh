#!/bin/bash

# Read each URL from url_list.txt and download the corresponding JSON file
while read -r url; do
    echo "Downloading $url..."
    curl -O "$url"
done < urls.txt
