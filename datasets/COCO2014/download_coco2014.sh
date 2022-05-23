#!/bin/bash

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

unzip -q train2014.zip
unzip -q val2014.zip

rm train2014.zip
rm val2014.zip

mkdir -p annotations
(cd annotations ;
wget --content-disposition https://polybox.ethz.ch/index.php/s/a7x8TgZFE3R1vcY/download
wget --content-disposition https://polybox.ethz.ch/index.php/s/Qu4238P6BJp6LBi/download
wget --content-disposition https://polybox.ethz.ch/index.php/s/sfrdSn6pBamlDGf/download)
