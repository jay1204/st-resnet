#!/bin/bash

# make a data folder
if ! [ -e data ]
then
    mkdir data
fi

pushd data

UCF_FLOW="ucf101_tvl1"
UCF_FLOW_ZIP="ucf101_tvl1.zip"
declare -a arr=("ucf101_tvl1_flow.zip.001" "ucf101_tvl1_flow.zip.002" "ucf101_tvl1_flow.zip.003")
if ! [ -d $UCF_FLOW ]
then
    echo $UCF_FLOW "not found, grab zip files"
    
    if ! [ -e $UCF_FLOW_ZIP ]
    then
        echo $UCF_FLOW_ZIP "not found downloading" 
        for i in "${arr[@]}"
        do 
            if ! [ -e $i ]
            then 
                echo $i "not found, downloading"
                wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/$i
            fi 
        done
        echo "Merge the parts into one zip file"
        cat "${arr[@]}" > $UCF_FLOW_ZIP
    fi
    echo "unzip file" $UCF_FLOW_ZIP
    unzip $UCF_FLOW_ZIP
fi

for i in "${arr[@]}"
do
    if [ -e $i ]
    then
        echo "remove" $i
        rm $i
    fi
done

if [ -e $UCF_FLOW_ZIP ]
then
    echo "remove" $UCF_FLOW_ZIP
    rm $UCF_FLOW_ZIP
fi

popd


