# Bar chart plotting wrapper
#   to plot a bar chart
#   with colours for each group
#   to plot precip bias


import numpy as np
import matplotlib.pyplot as plt
import sys,os
cwd=os.getcwd()
sys.path.append(cwd+'/..')
sys.path.append(cwd+'/../MetBot')
sys.path.append(cwd+'/../RTools')
sys.path.append(cwd+'/../quicks')
import MetBot.SynopticAnatomy as sy
import MetBot.EventStats as stats
import MetBot.SynopticPlot as syp
import MetBot.AdvancedPlots as ap
import MetBot.RainStats as rs
import MetBot.MetBlobs as blb
import MetBot.mytools as my
import MetBot.mynetcdf as mync
import glob, socket, os
import mpl_toolkits.basemap as bm
import MetBot.dset_dict as dsetdict
import MetBot.find_saddle as fs
import dsets_mplot_5group_4plot as dset_mp



### Running options
inorder=True    # to put the bars in order of number of TTTs
numlab=False     # to include number of TTTs in the yaxis label
group=True
globv='pr'
sub='contsub_nh'

weightlats=True


### What are we plotting?

# season
seas='NDJFM'

### Directories
bkdir=cwd+"/../../CTdata/"
botdir=bkdir+"metbot_multi_dset/"
thisdir=bkdir+"hpaperplot/"
refkey='0'

figdir=thisdir+"barcharts_rainbias/"
my.mkdir_p(figdir)


### Get season info
if seas == 'NDJFM':
    mons = [1, 2, 3, 11, 12]
    mon1 = 11
    mon2 = 3
    nmon = 5
elif seas == 'DJF':
    mons = [1, 2, 12]
    mon1 = 12
    mon2 = 2
    nmon = 3
elif seas == 'JF':
    mons = [1, 2]
    mon1 = 1
    mon2 = 2
    nmon = 2
elif seas=='all':
    mons=[1,2,3,4,5,6,7,8,9,10,11,12]
    nmon=12

### Dsets
dsets='all'
dsetnames=['cmip5']
ndset=len(dsetnames)
ndstr=str(ndset)

### Count total number of models
nm_dset=np.zeros(ndset)
for d in range(ndset):
    dset=dsetnames[d]
    nmod=len(dset_mp.dset_deets[dset])
    nm_dset[d]=nmod
nallmod=np.sum(nm_dset)
nallmod=int(nallmod)
print nallmod

if group:
    grcls=['fuchsia','gold','darkblue','r','blueviolet','springgreen']

### Open arrays for results
yvals = np.ma.zeros(nallmod, dtype=np.float32)
cols=["" for x in range(nallmod)]
modnm=["" for x in range(nallmod)] # creates a list of strings for modnames
cnt = 0

### First get reference data
refdset='trmm'
refmod='trmm_3b42v7'
refmoddct = dsetdict.dset_deets[refdset][refmod]
vnamedict = globv + 'name'
varstr = refmoddct[vnamedict]
ys = refmoddct['yrfname']

# Open ltmonmean file
meanfile = bkdir + 'metbot_multi_dset/' + refdset + '/' + refmod + '/' \
           + refmod + '.' + globv + '.mon.mean.' + ys + '.nc'

ncout = mync.open_multi(meanfile, globv, refmod, \
                        dataset=refdset, subs=sub)

ndim = len(ncout)
if ndim == 5:
    meandata, time, lat, lon, dtime = ncout
elif ndim == 6:
    meandata, time, lat, lon, lev, dtime = ncout
    meandata = np.squeeze(meandata)
dtime[:, 3] = 0

nlat=len(lat)
nlon=len(lon)

# Remove duplicate timesteps
print 'Checking for duplicate timesteps'
tmp = np.ascontiguousarray(dtime).view(
    np.dtype((np.void, dtime.dtype.itemsize * dtime.shape[1])))
_, idx = np.unique(tmp, return_index=True)
dtime = dtime[idx]
meandata = meandata[idx, :, :]

# Select seasons and get mean
thesemons = np.zeros((nmon, nlat, nlon), dtype=np.float32)
for zz in range(len(mons)):
    thesemons[zz, :, :] = meandata[mons[zz] - 1, :, :]
seasmean = np.nanmean(thesemons, 0)

# Regional mean
if weightlats:
    latr = np.deg2rad(lat)
    weights = np.cos(latr)
    zonmean = np.nanmean(seasmean, axis=1)
    reg_ref_mean = np.ma.average(zonmean, weights=weights)
else:
    reg_ref_mean = np.nanmean(seasmean)

print "Looping datasets"
for d in range(ndset):
    dset=dsetnames[d]
    dcnt=str(d+1)
    print 'Running on '+dset
    print 'This is dset '+dcnt+' of '+ndstr+' in list'

    ### Models
    mods = 'all'
    nmod = len(dset_mp.dset_deets[dset])
    mnames = list(dset_mp.dset_deets[dset])
    nmstr = str(nmod)

    for mo in range(nmod):
        name = mnames[mo]
        mcnt = str(mo + 1)
        print 'Running on ' + name
        print 'This is model ' + mcnt + ' of ' + nmstr + ' in list'

        ### if group get group
        if group:
            groupdct = dset_mp.dset_deets[dset][name]
            thisgroup = int(groupdct['group'])
            grcl = grcls[thisgroup - 1]
            cols[cnt]=grcl

        # Get details
        moddct = dsetdict.dset_deets[dset][name]

        ### Location for input & outputs
        indir = botdir + dset + "/"
        outdir = indir + name + "/"
        sysuf = outdir + name + '_'
        cal = moddct['calendar']
        ys = moddct['yrfname']

        ### Open rain data - historical
        print 'Getting historical rain data'
        globp = 'pr'
        if dset == 'noaa':
            raindset = 'trmm'
            rainmod = 'trmm_3b42v7'
            rmoddct = dsetdict.dset_deets[raindset][rainmod]
            rcal = rmoddct['calendar']
            rys = rmoddct['yrfname']
        else:
            raindset = dset
            rainmod = name
            rmoddct = moddct
            rcal = cal
            rys = ys

        rainname = rmoddct['prname']
        rainfile=botdir + raindset + "/" + rainmod + "/"+rainmod+"." + globp + ".mon.mean." + rys + ".nc"
        print 'Opening ' + rainfile

        rainout = mync.open_multi(rainfile, globp, rainmod, \
                                  dataset=raindset, subs=sub)

        rdim = len(rainout)
        if rdim == 5:
            rain, rtime, rlat, rlon, rdtime = rainout
        elif rdim == 6:
            rain, rtime, rlat, rlon, rlev, rdtime = rainout
            rain = np.squeeze(rain)
        else:
            print 'Check number of levels in ncfile'
        rdtime[:, 3] = 0
        nlat = len(rlat)
        nlon = len(rlon)

        print 'Checking for duplicate timesteps'  # do retain this - IPSL A LR has double tsteps
        tmp = np.ascontiguousarray(rdtime).view(np.dtype((np.void, rdtime.dtype.itemsize * rdtime.shape[1])))
        _, idx = np.unique(tmp, return_index=True)
        rdtime = rdtime[idx]
        rain = rain[idx, :, :]

        # Select seasons and get mean
        thesemons = np.zeros((nmon, nlat, nlon), dtype=np.float32)
        for zz in range(len(mons)):
            thesemons[zz, :, :] = rain[mons[zz] - 1, :, :]
        seasmean = np.nanmean(thesemons, 0)

        # Regional mean
        if weightlats:
            latr = np.deg2rad(rlat)
            weights = np.cos(latr)
            zonmean = np.nanmean(seasmean, axis=1)
            reg_mean = np.ma.average(zonmean, weights=weights)
        else:
            reg_mean = np.nanmean(seasmean)

        # Get bias
        bias= reg_mean - reg_ref_mean

        yvals[cnt] = bias

        ### Put name into string list
        modnm[cnt]=name
        cnt+=1

figsuf=sub
if group:
    figsuf=figsuf+'_grouped'

val=yvals[:]

# Open text file for results
file = open(figdir+'txtout.bias_'+globv+'.seas_'+seas+\
           '.'+figsuf+'.txt', "w")

if inorder:
    indsort=np.argsort(val)
    val4plot=val[indsort]
    mod4plot=[modnm[i] for i in indsort]
    col4plot=[cols[i] for i in indsort]
else:
    val4plot=val
    mod4plot=modnm
    col4plot=cols

pos=np.arange(nallmod)+0.5

modlabels=["" for x in range(nallmod)]
for m in range(nallmod):
    tmp=round(val4plot[m],2)
    strval=str(tmp)
    if numlab:
        modlabels[m]=mod4plot[m]+' ('+strval+')'
    else:
        modlabels[m] = mod4plot[m]
    file.write(mod4plot[m]+"\t"+str(int(val4plot[m]))+"\n")


plt.figure()
plt.subplots_adjust(left=0.3,right=0.9,top=0.9,bottom=0.1)
plt.barh(pos,val4plot, color=col4plot, edgecolor=col4plot, align='center')
plt.ylim(0,nallmod)
plt.yticks(pos,modlabels,fontsize=10, fontweight='demibold')
plt.xticks(np.arange(-1, 3.0, 0.5), fontsize=10, fontweight='demibold')
plt.xlabel('Precipitation Biases', fontsize=12, fontweight='demibold')

barfig=figdir+'/Barchart.bias_'+globv+'.seas_'+seas+'.'\
           +figsuf+'.png'
plt.savefig(barfig,dpi=150)
file.close()