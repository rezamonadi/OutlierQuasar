#!/bin/bash

cd spectra
for i in {0..7} 
do
    rsync --info=progress2 -h --no-motd --files-from=speclist-$i.txt rsync://data.sdss.org/dr16/eboss/spectro/redux/ . 2> /dev/null
done