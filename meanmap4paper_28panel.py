# To plot all CMIP5 models in multi-panel plot
# Longterm mean
# using new netcdf files

# Aiming at variables which I want to plot as a contour
# ctyp=abs
# ctyp=anom_mon - Composites plotted as anomaly from lt monthly mean (monthly mean for the month of each day selected)
# Now with option to add stippling if many days in composite agree
# Now with option to do biases

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import cm
import scipy.interpolate as spi

cwd=os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd+'/../MetBot')
sys.path.append(cwd+'/../RTools')
import PlotTools as pt
import MetBot.dset_dict as dsetdict
import dsets_mplot_5group_4plot as dset_mp
import MetBot.mast_dset_dict as mast_dict
import MetBot.dimensions_dict as dim_exdict
import MetBot.mytools as my
import MetBot.mynetcdf as mync

### Running options
xplots = 4
yplots = 7
seas='DJF'
spec_col=True
bias=False
biasper=False
group=True
varlist=['olr']
levsel=False
#levsel=False
if levsel:
    choosel=['700'] # can add a list
else:
    choosel=['1']
domain='aftrop'
if domain=='polar':
    sub='SH'
elif domain=='swio':
    sub='SA'
    figdim=[9,11]
elif domain=='aftrop':
    sub='AFTROP'
    figdim=[9,11]
elif domain=='nglob':
    sub='bigtrop'
    figdim=[11,9]
elif domain=='mac_wave':
    sub='SASA'
    figdim=[9,9]

if group:
    grcls=['fuchsia','gold','darkblue','r','blueviolet','springgreen']


### Get directories
bkdir=cwd+"/../../CTdata/"
thisdir=bkdir+"hpaperplot/"
botdir=bkdir+"metbot_multi_dset/"

figdir=thisdir+"allCMIPplot_ltmean/"
my.mkdir_p(figdir)

# variable
v=0
globv = varlist[v]
print "Running on "+globv

# levels - if levsel is false this will just be 1 level
l=0
print "Running on "+globv+" at pressure level "+choosel[l]

if seas == 'NDJFM':
    mons=[1,2,3,11,12]
    mon1 = 11
    mon2 = 3
    nmon = 5

if seas == 'DJF':
    mons=[1,2,12]
    mon1 = 12
    mon2 = 2
    nmon = 3

if seas == 'DJ':
    mons=[1,12]
    mon1 = 12
    mon2 = 1
    nmon = 2

if seas == 'N':
    mons=[11]
    mon1 = 11
    mon2 = 11
    nmon = 1

# Set up plot
print "Setting up plot..."
g, ax = plt.subplots(figsize=figdim)

cnt = 1

### Dsets
dsets='all'
ndset=len(dset_mp.dset_deets)
#dsetnames=list(dset_mp.dset_deets)
dsetnames=['noaa','cmip5']
ndstr=str(ndset)

print "Looping datasets"
for d in range(ndset):
    dset=dsetnames[d]
    dcnt=str(d+1)
    print 'Running on '+dset
    print 'This is dset '+dcnt+' of '+ndstr+' in list'

    if dset != 'cmip5': levc = int(choosel[l])
    else: levc = int(choosel[l]) * 100

    ### Models
    mods = 'all'
    nmod = len(dset_mp.dset_deets[dset])
    mnames_tmp = list(dset_mp.dset_deets[dset])
    nmstr = str(nmod)

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

    for mo in range(nmod):
        name = mnames[mo]
        mcnt = str(mo + 1)
        print 'Running on ' + name
        print 'This is model ' + mcnt + ' of ' + nmstr + ' in list'

        if group:
            groupdct = dset_mp.dset_deets[dset][name]
            thisgroup = int(groupdct['group'])
            grcl = grcls[thisgroup - 1]

        # Switch variable if NOAA
        if dset == 'noaa' and globv != 'olr':
            if globv == 'pr':
                ds4noaa = 'trmm'
                mod4noaa = 'trmm_3b42v7'
                #ds4noaa = 'ncep'
                #mod4noaa = 'ncep2'	
            else:
                # ds4noaa = 'ncep'
                # mod4noaa = 'ncep2'
                ds4noaa='era'
                mod4noaa='erai'
            dset2 = ds4noaa
            name2 = mod4noaa
        else:
            dset2 = dset
            name2 = name

        # Get info
        moddct = dsetdict.dset_deets[dset2][name2]
        vnamedict = globv + 'name'
        mastdct = mast_dict.mast_dset_deets[dset2]
        varstr = mastdct[vnamedict]
        dimdict = dim_exdict.dim_deets[globv][dset2]
        latname = dimdict[1]
        lonname = dimdict[2]
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

        if os.path.exists(meanfile):

            print 'Opening ' + meanfile

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
            if domain == 'mac_wave' or domain == 'bigtrop':
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

        else:
            print 'NO MEAN FILE AVAILABLE for ' + dset2 + '_' + name2
            print 'Setting mean to zero'
            meandata=np.zeros((12,nlat,nlon), dtype=np.float32)

        nlat=len(lat)
        nlon=len(lon)

        # Select seasons and get mean
        thesemons=np.zeros((nmon,nlat,nlon), dtype=np.float32)
        for zz in range(len(mons)):
            thesemons[zz,:,:]=meandata[mons[zz]-1,:,:]
        seasmean=np.nanmean(thesemons,0)

        if cnt == 1:
            m, f = pt.AfrBasemap(lat, lon, drawstuff=True, prj='cyl', fno=1, rsltn='l')
            if bias:
                reflon = lon[:]
                reflat = lat[:]
                refseas = seasmean[:]
                refnlon = nlon
                refnlat = nlat
        else:
            if bias:
                # Interpolate data
                prodata = np.zeros((refnlat, refnlon), dtype=np.float32)
                nonan_raw = np.nan_to_num(seasmean)
                Interpolator = spi.interp2d(lon, lat, nonan_raw[:, :], kind='linear')
                prodata[:, :] = Interpolator(reflon, reflat)

		if biasper:
		    seasmean= (prodata - refseas)/refseas*100.0
		else:
		    seasmean= prodata - refseas
                lon = reflon[:]
                lat = reflat[:]

        # Get lon lat grid
        plon, plat = np.meshgrid(lon, lat)

        data4plot=seasmean


        # Plot
        print "Plotting for model "+name2
        plt.subplot(yplots,xplots,cnt)
        if spec_col:
            if globv == 'olr':
                clevs = np.arange(200, 280, 10)
                cm = plt.cm.gray_r
            elif globv=='omega':
                if choosel[l]=='500':
                    clevs = np.arange(-0.10, 0.11, 0.01)
                elif choosel[l]=='200':
                    clevs = np.arange(-0.08, 0.088, 0.008)
                elif choosel[l]=='700':
                    clevs = np.arange(-0.10, 0.11, 0.01)
                cm = plt.cm.bwr
            elif globv=='pr':
                if bias:
                    if cnt==1:
                        clevs = np.arange(0,16,2)
                        cm = plt.cm.magma
                    else:
                        if biasper:
                            clevs= np.arange(-100.0,120.0,20)
                        else:
                            clevs= np.arange(-6.0,7.0,1)
                        cm = plt.cm.bwr_r
                else:
                    clevs = np.arange(0, 16, 2)
                    #cm = plt.cm.YlGnBu
                    cm = plt.cm.magma
            cs = m.contourf(plon, plat, data4plot, clevs, cmap=cm, extend='both')
        else:
            cs = m.contourf(plon, plat, data4plot, extend='both')

        plt.title(name2,fontsize=8, fontweight='demibold')

        # Redraw map
        m.drawcountries()
        m.drawcoastlines()
        if group:
            m.drawmapboundary(color=grcl, linewidth=3)

        cnt += 1

print "Finalising plot..."
plt.subplots_adjust(left=0.05,right=0.9,top=0.95,bottom=0.02,wspace=0.1,hspace=0.2)

# Plot cbar
axcl = g.add_axes([0.91, 0.15, 0.01, 0.6])
cbar = plt.colorbar(cs, cax=axcl)
my.ytickfonts(fontsize=12.)


if bias:
    if biasper:
	figsuf='biasper.'
    else:
	figsuf='bias.'
else:
    figsuf=''
if group:
    figsuf=figsuf+'_grouped.'

figname = figdir + 'multi_model_ltmean.'+figsuf + globv + \
          '.'+choosel[l]+'.'+sub+'.'+seas+'.png'
print 'saving figure as '+figname
plt.savefig(figname, dpi=150)
plt.close()