# Plotting wrapper for meridional mean OLR
#
#  Designed to be flexible to dataset
# and run on multiple models in a loop
# input at the top
# .....dset: noaa, um, cmip5, ncep, era, 20cr
# .....name: noaa or cdr, $mo_runid (e.g. anqjn), $cmip5_model_name, $reanal_name
# .....directory: here ../../CTdata/metbot_multi_dset/$dset/
# naming of ncfiles used here /$dset/$name.olr.day.mean.$firstyear_$lastyear.nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import sys,os
cwd=os.getcwd()
sys.path.append(cwd+'/..')
sys.path.append(cwd+'/../MetBot')
sys.path.append(cwd+'/../RTools')
sys.path.append(cwd+'/../quicks')
import MetBot.mynetcdf as mync
import MetBot.dset_dict as dsetdict
import MetBot.mast_dset_dict as mast_dict
import MetBot.mytools as my
import MetBot.dimensions_dict as dim_exdict


whichd='CMIP5' # UM or CMIP5

if whichd=='UM':
    import dsets_mplot_um as dset_mp
elif whichd=='CMIP5':
    import dsets_mplot_5group_4plot as dset_mp

### Running options
test_scr=False
sub="meridcross"
#sub="zonmada"
#sub="bigtrop"
seas="DJF"
future=False     # get future thresholds
globv='olr'
levsel=False
if levsel:
    choosel=['500'] # can add a list
else:
    choosel=['1']
l=0
group=True

### Directories
bkdir=cwd+"/../../CTdata/"
botdir=bkdir+"metbot_multi_dset/"
thisdir=bkdir+"hpaperplot/"


figdir=thisdir+"meridmean_conv/"
my.mkdir_p(figdir)

if future:
    fyear1='2065'
    fyear2='2099'

### Multi dset?
dsets='all'     # "all" or "spec" to choose specific dset(s)
if dsets=='all':
    if whichd=='UM':
        dsetnames=['noaa','cmip5','um']
    elif whichd=='CMIP5':
        dsetnames=['noaa','cmip5']
    ndset=len(dsetnames)
    dsetstr = '_'.join(dsetnames)
elif dsets=='spec': # edit for the dset you want
    ndset=1
    dsetnames=['noaa']
    dsetstr = '_'.join(dsetnames)
ndstr=str(ndset)
print 'Running on datasets:'
print dsetnames

### Count total number of models - (assumes using "all" models)
nm_dset=np.zeros(ndset)
for d in range(ndset):
    dset = dsetnames[d]
    nmod = len(dset_mp.dset_deets[dset])
    nm_dset[d]=nmod
nallmod=np.sum(nm_dset)
nallmod=int(nallmod)
print 'Total number of models = '+str(nallmod)

### Open array for names for cbar
modnm=["" for x in range(nallmod)] # creates a list of strings for modnames

### colours
if not group:
    cols=['b','g','r','c','m','gold','k',\
        'b','g','r','c','m','gold','k',\
        'b','g','r','c','m','gold','k',\
        'b','g','r','c','m','gold','k']
    markers=["o","o","o","o","o","o","o",\
        "^","^","^","^","^","^","^",\
        "*","*","*","*","*","*","*",\
        "d","d","d","d","d","d","d"]
    styls=["solid","solid","solid","solid","solid","solid","solid",\
        "dashed","dashed","dashed","dashed","dashed","dashed","dashed",\
        "dotted","dotted","dotted","dotted","dotted","dotted","dotted",\
        "-.","-.","-.","-.","-.","-.","-."]
elif group:
    grcls=['fuchsia','gold','darkblue','r','blueviolet','springgreen']
    grcnt=np.zeros(6,dtype=np.int8)
    grmrs=["o","^","*","d","+","v","h","o"]
    gstyls=["-","dotted","dashed","-.","-","dotted","dashed","-."]
lws = np.full((28), 2)
lws[0]=5
zorders = np.full((28), 2)
zorders[0]=3

if seas == 'DJF':
    mons=[1,2,12]
    mon1 = 12
    mon2 = 2
    nmon = 3

# Set up plot
print "Setting up plot..."
plt.figure(figsize=[10,6])
ax=plt.subplot(111)


if test_scr:
    ndset = 1

z=0
### Loop datasets
for d in range(ndset):
    dset=dsetnames[d]
    dcnt=str(d+1)
    print 'Running on '+dset
    print 'This is dset '+dcnt+' of '+ndstr+' in list'

    if dset != 'cmip5': levc = int(choosel[l])
    else: levc = int(choosel[l]) * 100

    ### Multi model?
    mods='all'  # "all" or "spec" to choose specific model(s)
    if mods=='all':
        nmod=len(dset_mp.dset_deets[dset])
        mnames_tmp=list(dset_mp.dset_deets[dset])
    nmstr=str(nmod)

    if dset == 'cmip5':
        if group:
            mnames = np.zeros(nmod, dtype=object)

            for mo in range(nmod):
                name = mnames_tmp[mo]
                moddct = dset_mp.dset_deets[dset][name]
                thisord = int(moddct['ord']) - 2  # minus 2 because cdr already used
                mnames[thisord] = name

        else:
            mnames = mnames_tmp
    else:
        mnames = mnames_tmp

    for m in range(nmod):
        name=mnames[m]
        mcnt=str(m+1)
        print 'Running on ' + name
        print 'This is model '+mcnt+' of '+nmstr+' in list'

        if group:
            groupdct = dset_mp.dset_deets[dset][name]
            thisgroup = int(groupdct['group'])
            grcl = grcls[thisgroup - 1]
            grmr=grmrs[grcnt[thisgroup-1]]
            grstl=gstyls[grcnt[thisgroup-1]]
            grcnt[thisgroup-1]+=1

        # Switch variable if NOAA
        if dset == 'noaa' and globv != 'olr':
            if globv == 'pr':
                ds4noaa = 'trmm'
                mod4noaa = 'trmm_3b42v7'
                #ds4noaa = 'ncep'
                #mod4noaa = 'ncep2'
            else:
                ds4noaa = 'ncep'
                mod4noaa = 'ncep2'
            dset2 = ds4noaa
            name2 = mod4noaa
        else:
            dset2 = dset
            name2 = name

        # Get details
        moddct=dsetdict.dset_deets[dset2][name2]
        vnamedict = globv + 'name'
        mastdct = mast_dict.mast_dset_deets[dset2]
        varstr = mastdct[vnamedict]
        dimdict = dim_exdict.dim_deets[globv][dset2]
        latname = dimdict[1]
        lonname = dimdict[2]
        if dset2=='um':
            moddct1 = dset_mp.dset_deets[dset2][name2]
            ys=moddct1['climyr']
        else:
            if globv != 'omega' and globv != 'q' and globv != 'gpth':
                ys = moddct['yrfname']
            else:
                if name2 == "MIROC5":
                    if globv == 'q':
                        ys = moddct['fullrun']
                    elif globv == 'omega' or globv == 'gpth':
                        ys = '1950_2009'
                    else:
                        print 'variable ' + globv + ' has unclear yearname for ' + name2
                else:
                    ys = moddct['fullrun']

        # Open ltmonmean file
        meanfile = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
                   + name2 + '.' + globv + '.mon.mean.' + ys + '.nc'

        print meanfile

        if os.path.exists(meanfile):

            if levsel:
                ncout = mync.open_multi(meanfile, globv, name2, \
                                        dataset=dset2, subs=sub, levsel=levc)
            else:
                ncout = mync.open_multi(meanfile, globv, name2, \
                                        dataset=dset2, subs=sub)
            print '...file opened'
            ndim = len(ncout)
            if ndim == 5:
                meandata, time, lat, lon, dtime = ncout
            elif ndim == 6:
                meandata, time, lat, lon, lev, dtime = ncout
                meandata = np.squeeze(meandata)
            else:
                print 'Check number of dims in ncfile'
            dtime[:, 3] = 0

            # Fix lat and lons if it spans 0
            if sub == 'bigtrop':
                print "Ammending lons around 0"
                for i in range(len(lon)):
                    if lon[i] > 180:
                        lon[i] = lon[i] - 360
                ord = np.argsort(lon)
                lon = lon[ord]
                meandata = meandata[:, :, ord]

            # Remove duplicate timesteps
            print 'Checking for duplicate timesteps'
            tmp = np.ascontiguousarray(dtime).view(
                np.dtype((np.void, dtime.dtype.itemsize * dtime.shape[1])))
            _, idx = np.unique(tmp, return_index=True)
            dtime = dtime[idx]
            meandata = meandata[idx, :, :]

            nlat = len(lat)
            nlon = len(lon)

            # Select seasons and get mean
            thesemons = np.zeros((nmon, nlat, nlon), dtype=np.float32)
            for zz in range(len(mons)):
                thesemons[zz, :, :] = meandata[mons[zz] - 1, :, :]
            seasmean = np.nanmean(thesemons, 0)

            # Get meridional mean
            meridmean=np.nanmean(seasmean,0)

            if group:
                colour = grcl
                mk = grmr
                ls= grstl
            else:
                colour = cols[z]
                mk = markers[z]
                ls= styls[z]

            if name=='cdr':
                colour='k'

            lw=lws[z]
            zord=zorders[z]

            #if thisgroup==7 or name=='cdr':

            plt.plot(lon,meridmean,c=colour,linestyle=ls, linewidth=lw, zorder=zord,label=name,)

            ### Put name into string list
            modnm[z] = dset + "_" + name

        else:
            print 'No file for model '+name

        z += 1

        print 'Finished running on ' + name
        print 'This is model '+mcnt+' of '+nmstr+' in list'



### Plot legend and axis
plt.xlim(0,100)
plt.xlabel('longitude', fontsize=10.0, weight='demibold', color='k')
plt.ylabel('meridional mean OLR', fontsize=10.0, weight='demibold', color='k')

if globv=='olr':
    plt.ylim(180,290)

box=ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.legend(loc='center left',bbox_to_anchor=[1,0.5],fontsize='xx-small')

figsuf = whichd

if group:
    figsuf=figsuf+'_grouped'

if test_scr:
    figsuf=figsuf+'_test_scr'

### Save figure
figname=figdir+'ZonalMean.'+globv+'.'+seas+'.'+sub+'.'+figsuf+'.png'
plt.savefig(figname)
print 'Saving figure as '+figname

plt.close('all')
