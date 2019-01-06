#!/bin/bash

# Script to interpolate netcdf files of long term monthly means for all variables
# and output to a new netcdf file
# To help with getting quick composite plots

# Find input and output directories
alldir=../../../CTdata
mbdir=$alldir/metbot_multi_dset
dset_dict=../../quicks/dicts4CDO/dset_info_4CDO.28models.txt

res=360x180

# Loop datasets - by name because they are original
echo 'Looping datasets and models'
#for name in MIROC-ESM;do
#for name in $(more $dset_dict | gawk '{print $1}');do
for name in cdr;do
    echo $name
    dset=$(grep -w $name $dset_dict  | gawk '{print $2}' | head -1)
    echo $dset

    #Loop variables
    echo 'Looping variables'
    for var in olr;do
    #for var in olr pr omega gpth u v q T;do
    #for var in pr omega gpth u v q T;do
    #for var in olr pr;do

        echo 'Running on'
        echo $var

        dict=../../quicks/dicts4CDO/dset_info_4CDO.28models.$var.txt
        name2=$(grep -w $name $dict | gawk '{print $3}' | head -1)
        dset2=$(grep -w $name $dict | gawk '{print $4}' | head -1)

        indir=$mbdir/$dset2
        outdir=$indir/$name2

        if [ "$dset2" == "trmm" ]; then
            year1=1998
            year2=2013
        else
            if [ "$dset2" == "cmip5" ]; then
                year1=1970
                year2=2004
            else
                year1=1979
                year2=2013
            fi
        fi

        outfile1=$outdir/$name2.$var.mon.mean.${year1}_${year2}.nc
        outfile2=$outdir/$name2.$var.mon.mean.${year1}_${year2}.remap_${res}.nc

        cdo remapcon,r${res} $outfile1 $outfile2

    done
done