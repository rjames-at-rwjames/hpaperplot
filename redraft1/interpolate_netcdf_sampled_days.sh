#!/bin/bash

# Script to interpolate netcdf file of sampled dates to a new netcdf files

# Find input and output directories
alldir=../../../CTdata
mbdir=$alldir/metbot_multi_dset
dset_dict=../../quicks/dicts4CDO/dset_info_4CDO.28models.txt

res=360x180

frmevnt=first # Option to only use first day from each event - 'first' or 'all'
thname=actual # Option to use other thresholds (would need to make new txtfiles first
lag=True
for sample in blon;do # blat, blon, bang, blon2

#if [ "$sample" == "blon" ] ; then
#	frmevnt=first
#else
#	frmevnt=all
#fi
echo $sample
echo $frmevnt

# Loop continental and madagascan samples
echo 'Looping continental and madagascan'
for wcb in cont;do

    # Loop datasets - by name because they are original
    echo 'Looping datasets and models'

#    for name in ;do
    for name in $(more $dset_dict | gawk '{print $1}');do
        echo $name
        dset=$(grep -w $name $dset_dict  | gawk '{print $2}' | head -1)
        echo $dset

        #Loop variables
        echo 'Looping variables'
        for var in omega;do
        #for var in olr pr omega gpth u v q T;do # doesn't work well for TRMM because short timeperiod
#        for var in omega gpth u v q T pr;do # doesn't work well for TRMM because short timeperiod
    	#for var in u v;do

            echo "Running on"
            echo $var
            dict=../../quicks/dicts4CDO/dset_info_4CDO.28models.$var.txt
            name2=$(grep -w $name $dict | gawk '{print $3}' | head -1)
            dset2=$(grep -w $name $dict | gawk '{print $4}' | head -1)
            ysname=$(grep -w $name $dict | gawk '{print $5}' | head -1)

            indir=$mbdir/$dset2
            outdir=$indir/$name2

            if [ "$lag" != "True" ]; then

                outfile1=$outdir/$name.$name2.$var.sampled_days.$sample.$frmevnt.$wcb.$thname.nc
                outfile2=$outdir/$name.$name2.$var.sampled_days.$sample.$frmevnt.$wcb.$thname.remap_${res}.nc

                cdo remapcon,r${res} $outfile1 $outfile2

            else

                for eday in -3 -2 -1 0 1 2 3;do

                    outout=$outdir/lag_samples/

                    outfile1=$outout/$name.$name2.$var.sampled_days.$sample.$frmevnt.$wcb.$thname.lag_${eday}.nc
                    outfile2=$outout/$name.$name2.$var.sampled_days.$sample.$frmevnt.$wcb.$thname.lag_${eday}.remap_${res}.nc

                    cdo remapcon,r${res} $outfile1 $outfile2

                done # lag

            fi # lag

        done
    done
done
done