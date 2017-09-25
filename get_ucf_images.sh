#!/bin/bash

# make a data folder
if ! [ -e data ]
then
    mkdir data
fi

pushd data

UCF_IMAGE="jpegs_256"
UCF_IMAGE_ZIP="ucf101_jpegs.zip"
declare -a arr=("ucf101_jpegs_256.zip.001" "ucf101_jpegs_256.zip.002" "ucf101_jpegs_256.zip.003")
if ! [ -d $UCF_IMAGE ]
then
    echo $UCF_IMAGE "not found, grab zip files"
    
    if ! [ -e $UCF_IMAGE_ZIP ]
    then
        echo $UCF_IMAGE_ZIP "not found downloading" 
        for i in "${arr[@]}"
        do 
            if ! [ -e $i ]
            then 
                echo $i "not found, downloading"
                wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/$i
            fi 
        done
        echo "Merge the parts into one zip file"
        cat "${arr[@]}" > $UCF_IMAGE_ZIP
    fi
    echo "unzip file" $UCF_IMAGE_ZIP
    unzip $UCF_IMAGE_ZIP
fi

for i in "${arr[@]}"
do
    if [ -e $i ]
    then
        echo "remove" $i
        rm $i
    fi
done

if [ -e $UCF_IMAGE_ZIP ]
then
    echo "remove" $UCF_IMAGE_ZIP
    rm $UCF_IMAGE_ZIP
fi

popd


