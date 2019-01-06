# To test variation between model composites
# using ANOVA and Kruskal-Wallis (non-parametric)
# on each gridpoint

# Aiming at variables which I want to plot as a contour
# ctyp=abs
# ctyp=anom_mon

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import cm

cwd=os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd+'/../../MetBot')
sys.path.append(cwd+'/../../RTools')
sys.path.append(cwd+'/../')
import PlotTools as pt
import MetBot.dset_dict as dsetdict
import MetBot.mast_dset_dict as mast_dict
import MetBot.dimensions_dict as dim_exdict
import MetBot.mytools as my
import MetBot.mynetcdf as mync
import scipy
import scipy.interpolate as spi
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


### Running options
first_test=False
test_scr=False
remove_outliers=False
howmany='many' # 'two' or 'many'

runs=['opt1']
ctyps=['anom_seas'] #abs is absolute,  anom_mon is rt monthly mean, anom_seas is rt seasonal mean
wcb=['cont'] # which cloud band composite? Options: cont, mada, dbl
spec_col=True
varlist=['omega']
thname='actual'
levsel=True
if levsel:
    choosel=['500'] # can add a list
else:
    choosel=['1']
domain='swio'
lag=False
seas='NDJFM'
climyr='spec' # this is to use new climatology files which are based on only 35 years
                # 'spec' is 35 years
                # 'prev' is previous files - all different climatologies

## Interpolation options
interp='file' # 'file' (to import interpolated file) or 'here' to do it in this script
if interp=='file':
    fileres='360x180'
    res=1.0
elif interp=='here':
    res=1.0

## Info for options
if domain=='polar':
    sub='SH'
elif domain=='swio':
    sub='SA'
    figdim=[9,7]
    dom = ((-59.5,-0.5), (0.0, 100.0))
elif domain=='nglob':
    sub='bigtrop'
elif domain=='mac_wave':
    sub='SASA'
    figdim=[9,9]

if remove_outliers:
    if howmany=='two':
        import dsets_nooutlier as dset_mp
    elif howmany=='many':
        import dsets_bestfit as dset_mp
else:
    import dsets_paper_28_4plot as dset_mp

if lag:
    edays=[-3,-2,-1,0,1,2,3]
else:
    edays=[0]


### Get directories
bkdir=cwd+"/../../../CTdata/"
thisdir=bkdir+"/hpaperplot/"
botdir=bkdir+"metbot_multi_dset/"

compdir=thisdir+"comp4paper_anova/"
if lag:
    compdir=thisdir+"comp4paper_anova_lags/"
my.mkdir_p(compdir)

# levels - if levsel is false this will just be 1 level
l=0
print "Running at pressure level "+choosel[l]

if seas == 'NDJFM':
    mons=[1,2,3,11,12]
    mon1 = 11
    mon2 = 3
    nmon = 5

# Loop variables
for v in range(len(varlist)):
    globv = varlist[v]
    print "Running on "+globv


    # Loop sampling options
    for r in range(len(runs)):

        if runs[r]=='opt1':
            sample='blon'
            from_event='first'
        elif runs[r]=='opt2':
            sample='blon2'
            from_event='all'
        elif runs[r]=='opt3':
            sample='blon2'
            from_event='first'

        # Loop abs and anom
        for a in range(len(ctyps)):
            ctyp=ctyps[a]
            print "Running on "+ctyp

            # Loop sample type
            for o in range(len(wcb)):
                type = wcb[o]

                # Loop lags
                for lo in range(len(edays)):
                    print "Running with a lag of "+str(edays[lo])

                    # Make a grid
                    lt1, lt2 = dom[0];ln1, ln2 = dom[1]
                    lats = np.arange(lt1, lt2 + 1, res)
                    lons = np.arange(ln1, ln2 + 1, res)
                    nlat=len(lats)
                    nlon=len(lons)

                    if first_test:
                        lonloop=5
                        latloop=5
                    else:
                        lonloop=nlon
                        latloop=nlat

                    # Generate arrays for the mapping
                    f_grid=np.zeros((nlat,nlon),dtype=np.float32)
                    pf_grid=np.zeros((nlat,nlon),dtype=np.float32)

                    k_grid=np.zeros((nlat,nlon),dtype=np.float32)
                    pk_grid=np.zeros((nlat,nlon),dtype=np.float32)

                    # Loop gridboxes
                    for i in range(latloop):
                        for j in range(lonloop):

                            # Set the model sample
                            dset='cmip5'
                            if test_scr:
                                nmod=3
                            else:
                                nmod = len(dset_mp.dset_deets[dset])
                            nmstr = str(nmod)
                            mnames = list(dset_mp.dset_deets[dset])

                            # Make array for models and composites
                            collect=[np.zeros(50,dtype=np.float32)]*nmod

                            if dset != 'cmip5':
                                levc = int(choosel[l])
                            else:
                                levc = int(choosel[l]) * 100

                            for mo in range(nmod):
                                name = mnames[mo]
                                mcnt = str(mo + 1)
                                print 'Running on ' + name
                                print 'This is model ' + mcnt + ' of ' + nmstr + ' in list'

                                # Switch variable if NOAA
                                if dset == 'noaa' and globv != 'olr':
                                    if globv == 'pr':
                                        ds4noaa = 'trmm'
                                        mod4noaa = 'trmm_3b42v7'
                                    else:
                                        ds4noaa = 'ncep'
                                        mod4noaa = 'ncep2'
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



                                # Open sample file
                                inpath=bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/'
                                filend=''
                                if lag:
                                    filend=filend+'.lag_'+str(edays[lo])
                                    inpath=inpath+'lag_samples/'
                                if interp=='file':
                                    filend=filend+'.remap_'+fileres

                                smpfile=inpath+name+'.'+name2+'.'+globv+'.sampled_days.'\
                                    +sample+'.'+from_event+'.'+type+'.'+thname+filend+'.nc'

                                print 'Opening '+smpfile

                                if os.path.exists(smpfile):

                                    print 'File exists'

                                    if levsel:
                                        ncout = mync.open_multi(smpfile, globv, name2, \
                                                                dataset=dset2, subs=sub, levsel=levc)
                                    else:
                                        ncout = mync.open_multi(smpfile, globv, name2, \
                                                        dataset=dset2,subs=sub)
                                    print '...file opened'
                                    ndim = len(ncout)
                                    if ndim == 5:
                                        smpdata, time, lat, lon, smpdtime = ncout
                                    elif ndim == 6:
                                        smpdata, time, lat, lon, lev, smpdtime = ncout
                                        smpdata = np.squeeze(smpdata)
                                    else:
                                        print 'Check number of dims in ncfile'
                                    smpdtime[:, 3] = 0

                                    # Fix lat and lons if it spans 0
                                    if domain == 'mac_wave' or domain == 'bigtrop':
                                        print "Ammending lons around 0"
                                        for i in range(len(lon)):
                                            if lon[i] > 180:
                                                lon[i] = lon[i] - 360
                                        ord = np.argsort(lon)
                                        lon = lon[ord]
                                        smpdata = smpdata[:, :, ord]

                                    # Remove duplicate timesteps
                                    print 'Checking for duplicate timesteps'
                                    tmp = np.ascontiguousarray(smpdtime).view(np.dtype((np.void,smpdtime.dtype.itemsize * smpdtime.shape[1])))
                                    _, idx = np.unique(tmp, return_index=True)
                                    smpdtime= smpdtime[idx]
                                    smpdata=smpdata[idx,:,:]

                                    # Count timesteps
                                    nsamp=len(smpdata[:,0,0])
                                    this_nlat = len(lat)
                                    this_nlon = len(lon)


                                    # If anomaly open ltmean file
                                    if ctyp == 'anom_seas':

                                        # Years for clim
                                        if climyr == 'spec':
                                            ysclim = moddct['yrfname']
                                        else:
                                            ysclim = ys
                                        year1 = float(ysclim[0:4])
                                        year2 = float(ysclim[5:9])

                                        filend2=''
                                        if interp == 'file':
                                            filend2 = filend2 + '.remap_' + fileres

                                        # Open ltmonmean file
                                        meanfile = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
                                                   + name2 + '.' + globv + '.mon.mean.' + ysclim + filend2+'.nc'

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

                                        # get seasonal mean
                                        thesemons = np.zeros((nmon, this_nlat, this_nlon), dtype=np.float32)
                                        for zz in range(len(mons)):
                                            thesemons[zz, :, :] = meandata[mons[zz] - 1, :, :]
                                        seasmean = np.nanmean(thesemons, 0)

                                        anoms = np.asarray([smpdata[x, :, :] - seasmean for x in range(len(smpdata[:, 0, 0]))])

                                    if ctyp=='abs':
                                        chosedata=smpdata
                                    elif ctyp=='anom_seas':
                                        chosedata=anoms


                                    # Interpolate data
                                    if interp=='here':
                                        print "Interpolating data to a " + str(res) + " grid"
                                        prodata = np.zeros((nsamp, nlat, nlon), dtype=np.float32)

                                        # Get rid of nans
                                        nonan = np.nan_to_num(chosedata)

                                        for step in range(nsamp):
                                            Interpolator = spi.interp2d(lon, lat, nonan[step, :, :], kind='linear')
                                            prodata[step,:,:] = Interpolator(lons,lats)

                                    else:
                                        prodata=chosedata
                                        if dset=='cmip5':
                                            lat=lat[::-1]
                                            prodata=prodata[:,::-1,:]

                                    # Check smpdata
                                    #print 'Checking data'
                                    #print smpdata
                                    #print 'for this gridpoint'
                                    #print smpdata[:,i,j]

                                    # Select this gridpoint
                                    collect[mo]=prodata[:,i,j]
                                    print collect[mo]

                            # anova test for this gridpoint
                            print "Sample data collected for this gridpoint:"
                            print collect
                            print "Running ANOVA..."
                            f, p = scipy.stats.f_oneway(*collect)
                            print "F stat is:"
                            print f
                            f_grid[i,j] = f
                            print "p value is:"
                            print p
                            pf_grid[i,j] = p

                            "Running Kruskal-Wallis..."
                            K, pval = scipy.stats.kruskal(*collect)
                            print "Output stat is:"
                            print K
                            k_grid[i,j] = K
                            print "p value is:"
                            print pval
                            pk_grid[i,j] = pval

                    # Map values
                    print "Setting up plot..."
                    g, ax = plt.subplots(figsize=figdim)

                    m, f = pt.AfrBasemap(lats, lons, drawstuff=True, prj='cyl', fno=1, rsltn='l')
                    plon, plat = np.meshgrid(lons, lats)
                    xplots=2
                    yplots=2

                    # Plot fstat
                    print 'Plotting ANOVA statistic'
                    ax2 = plt.subplot(yplots,xplots,1)
                    cs = m.contourf(plon, plat, f_grid)
                    m.drawcountries()
                    m.drawcoastlines()
                    plt.title('ANOVA: F value')

                    cax = make_axes_locatable(ax2).append_axes("right", size="5%", pad="2%")
                    cb = plt.colorbar(cs, cax=cax)



                    # Plot pvalue for fstat
                    print 'Plotting ANOVA p value'
                    ax2 = plt.subplot(yplots,xplots,2)
                    clevs = np.arange(0,0.1,0.01)
                    cs = m.contourf(plon, plat, pf_grid, clevs, extend='both')
                    m.drawcountries()
                    m.drawcoastlines()
                    plt.title('ANOVA: p value')

                    cax = make_axes_locatable(ax2).append_axes("right", size="5%", pad="2%")
                    cb = plt.colorbar(cs, cax=cax)


                    # Plot H stat
                    print 'Plotting Kruskal statistic'
                    ax2 = plt.subplot(yplots,xplots,3)
                    cs = m.contourf(plon, plat, k_grid)
                    m.drawcountries()
                    m.drawcoastlines()
                    plt.title('Kruskal-Wallis: H value')

                    cax = make_axes_locatable(ax2).append_axes("right", size="5%", pad="2%")
                    cb = plt.colorbar(cs, cax=cax)



                    # Plot fstat
                    print 'Plotting p value for H statistic'
                    ax2 = plt.subplot(yplots,xplots,4)
                    clevs = np.arange(0,0.1,0.01)
                    cs = m.contourf(plon, plat, pk_grid, clevs, extend='both')
                    m.drawcountries()
                    m.drawcoastlines()
                    plt.title('Kruskal-Wallis: p value')

                    cax = make_axes_locatable(ax2).append_axes("right", size="5%", pad="2%")
                    cb = plt.colorbar(cs, cax=cax)


                    if test_scr:
                        mods='testmodels'
                    elif remove_outliers:
                        mods='nooutliers_'+howmany
                    else:
                        mods='allmod'

                    compname= compdir + 'statsplot_test.'+globv+'.'+choosel[l]+'.'+ctyp+'.interp_'+interp+'_'+str(res)+'.models_'+mods+'.png'
                    plt.savefig(compname, dpi=150)
                    plt.close()