# To test the similarity of each model with reference
# on a gridpoint by gridpoint basis
# for composites

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
import dsets_paper_28_4plot as dset_mp
import MetBot.mast_dset_dict as mast_dict
import MetBot.dimensions_dict as dim_exdict
import MetBot.mytools as my
import MetBot.mynetcdf as mync
import MetBot.MetBlobs as blb
import scipy

### Running options
test_scr=False
xplots = 4
yplots = 7
alphord=True
runs=['opt1']
ctyp='anom_seas' #anom_seas is rt seasonal mean
wcb=['cont'] # which cloud band composite? Options: cont, mada, dbl
spec_col=True
varlist=['olr']
thname='actual'
levsel=False
if levsel:
    choosel=['500'] # can add a list
else:
    choosel=['1']
domain='swio'
if domain=='polar':
    sub='SH'
elif domain=='swio':
    sub='SA'
    figdim=[9,11]
elif domain=='nglob':
    sub='bigtrop'
elif domain=='mac_wave':
    sub='SASA'
    figdim=[9,9]
lag=False
if lag:
    edays=[-3,-2,-1,0,1,2,3]
else:
    edays=[0]

fdr=False

## Interpolation options
interp='file' # 'file' (to import interpolated file) or 'here' to do it in this script
if interp=='file':
    fileres='360x180'
    res=1.0
elif interp=='here':
    res=1.0

seas='NDJFM'
climyr='spec' # this is to use new climatology files which are based on only 35 years
                # 'spec' is 35 years
                # 'prev' is previous files - all different climatologies

### Get directories
bkdir=cwd+"/../../../CTdata/"
thisdir=bkdir+"/hpaperplot/"
botdir=bkdir+"metbot_multi_dset/"

compdir=thisdir+"comp4paper_t_test/"
if lag:
    compdir=thisdir+"comp4paper_t_test_lags/"
my.mkdir_p(compdir)

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

    # Loop sample type
    for o in range(len(wcb)):
        type = wcb[o]

        # Loop lags
        for lo in range(len(edays)):
            print "Running with a lag of "+str(edays[lo])

            # First open the file for CDR
            print 'Opening reference data'
            dset='noaa'
            name='cdr'
            ysclim = '1979_2013'
            levc = int(choosel[l]) * 100

            if globv=='olr':
                dset2=dset
                name2=name

            inpath = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/'
            filend = ''
            if lag:
                filend = filend + '.lag_' + str(edays[lo])
                inpath = inpath + 'lag_samples/'
            if interp == 'file':
                filend = filend + '.remap_' + fileres

            smpfile = inpath + name + '.' + name2 + '.' + globv + '.sampled_days.' \
                      + sample + '.' + from_event + '.' + type + '.' + thname + filend + '.nc'

            print 'Opening reference file for sample'
            print smpfile

            if levsel:
                ncout = mync.open_multi(smpfile, globv, name2, \
                                        dataset=dset2, subs=sub, levsel=levc)
            else:
                ncout = mync.open_multi(smpfile, globv, name2, \
                                        dataset=dset2, subs=sub)
            print '...file opened'
            ndim = len(ncout)
            if ndim == 5:
                refsmp, time, lat, lon, smpdtime = ncout
            elif ndim == 6:
                refsmp, time, lat, lon, lev, smpdtime = ncout
                refsmp = np.squeeze(refsmp)
            else:
                print 'Check number of dims in ncfile'
            smpdtime[:, 3] = 0

            # Count timesteps
            nsamp = len(refsmp[:, 0, 0])
            this_nlat = len(lat)
            this_nlon = len(lon)

            # Years for clim
            year1 = float(ysclim[0:4])
            year2 = float(ysclim[5:9])

            filend2 = ''
            if interp == 'file':
                filend2 = filend2 + '.remap_' + fileres

            # Open ltmonmean file
            meanfile = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
                       + name2 + '.' + globv + '.mon.mean.' + ysclim + filend2 + '.nc'

            print 'Opening climatology file for reference'
            print meanfile

            if levsel:
                ncout = mync.open_multi(meanfile, globv, name2, \
                                        dataset=dset2, subs=sub, levsel=levc)
            else:
                ncout = mync.open_multi(meanfile, globv, name2, \
                                        dataset=dset2, subs=sub)
            print '...file opened'
            ndim = len(ncout)
            if ndim == 5:
                refmean, time, lat, lon, dtime = ncout
            elif ndim == 6:
                refmean, time, lat, lon, lev, dtime = ncout
                refmean = np.squeeze(meandata)
            else:
                print 'Check number of dims in ncfile'
            dtime[:, 3] = 0

            # get seasonal mean
            print 'Calculating seasonal mean for reference'
            thesemons = np.zeros((nmon, this_nlat, this_nlon), dtype=np.float32)
            for zz in range(len(mons)):
                thesemons[zz, :, :] = refmean[mons[zz] - 1, :, :]
            refseasmean = np.nanmean(thesemons, 0)

            print 'Calculating anomalies for reference'
            refanoms = np.asarray([refsmp[x, :, :] - refseasmean for x in range(len(refsmp[:, 0, 0]))])

            # Open plotting
            print 'Setting up plot before looping models'
            g, ax = plt.subplots(figsize=figdim)

            cnt = 1

            print 'Looping models'

            ### Dsets
            dsets='spec'
            dsetnames=['cmip5']
            ndset=len(dsetnames)
            ndstr=str(ndset)

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
                    if alphord:
                        mnames=sorted(mnames_tmp,key=lambda s: s.lower())
                    else:
                        mnames = mnames_tmp
                else:
                    mnames = mnames_tmp

                if test_scr:
                    nmod=1

                for mo in range(nmod):
                    name = mnames[mo]
                    mcnt = str(mo + 1)
                    print 'Running on ' + name
                    print 'This is model ' + mcnt + ' of ' + nmstr + ' in list'

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

                    # Years for clim and manntest
                    if climyr == 'spec':
                        ysclim = moddct['yrfname']
                    else:
                        ysclim = ys
                    year1 = float(ysclim[0:4])
                    year2 = float(ysclim[5:9])

                    # Open sample file
                    inpath = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/'
                    filend = ''
                    if lag:
                        filend = filend + '.lag_' + str(edays[lo])
                        inpath = inpath + 'lag_samples/'
                    if interp == 'file':
                        filend = filend + '.remap_' + fileres

                    smpfile = inpath + name + '.' + name2 + '.' + globv + '.sampled_days.' \
                              + sample + '.' + from_event + '.' + type + '.' + thname + filend + '.nc'

                    print 'Opening sample file for model'
                    print 'Opening '+smpfile

                    if os.path.exists(smpfile):

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
                        nlat = len(lat)
                        nlon = len(lon)

                        filend2 = ''
                        if interp == 'file':
                            filend2 = filend2 + '.remap_' + fileres

                        # Open ltmonmean file
                        meanfile = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
                                   + name2 + '.' + globv + '.mon.mean.' + ysclim + filend2 + '.nc'

                        print 'Opening climatology file for model'
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

                        # get seasonal mean
                        thesemons = np.zeros((nmon, this_nlat, this_nlon), dtype=np.float32)
                        for zz in range(len(mons)):
                            thesemons[zz, :, :] = meandata[mons[zz] - 1, :, :]
                        seasmean = np.nanmean(thesemons, 0)

                        anoms = np.asarray([smpdata[x, :, :] - seasmean for x in range(len(smpdata[:, 0, 0]))])

                        # if dset == 'cmip5':
                        #     lat = lat[::-1]
                        #     anoms = anoms[:, ::-1, :]

                        # Test difference between this model and reference
                        print 'Starting test between '+name2+' and reference'

                        uvals = np.zeros((nlat, nlon), dtype=np.float32)
                        pvals = np.zeros((nlat, nlon), dtype=np.float32)
                        for i in range(nlat):
                            for j in range(nlon):

                                refbox = refanoms[:, i, j]
                                modbox = anoms[:, i, j]
                                ustat, pvalue = scipy.stats.mannwhitneyu(refbox, modbox, alternative='two-sided')

                                uvals[i, j] = ustat
                                pvals[i, j] = pvalue

                        print uvals
                        print pvals

                        if not fdr:
                            # Simple approach - Get mask over values where pval <= a specified value
                            mask_pvals = np.zeros((nlat, nlon), dtype=np.float32)
                            for i in range(nlat):
                                for j in range(nlon):
                                    if pvals[i, j] <= 0.01:
                                        mask_pvals[i, j] = 1
                                    else:
                                        mask_pvals[i, j] = 0

                        if cnt == 1:
                            m, f = pt.AfrBasemap(lat, lon, drawstuff=True, prj='cyl', fno=1, rsltn='l')

                        # Get lon lat grid
                        plon, plat = np.meshgrid(lon, lat)

                        anom_comp = np.nanmean(anoms, 0)

                        # Plot
                        print "Plotting for model "+name2
                        plt.subplot(yplots,xplots,cnt)
                        if spec_col:
                            if globv == 'olr':
                                clevs= np.arange(-40,50,10)
                                cm = plt.cm.BrBG_r
                            elif globv=='pr':
                                clevs = np.arange(-12,14, 2)
                                cm = plt.cm.bwr_r
                            elif globv == 'omega':
                                clevs = np.arange(-0.12, 0.14, 0.02)
                                cm = plt.cm.bwr
                            elif globv == 'q':
                                clevs = np.arange(-0.004, 0.0045, 0.0005)
                                cm = plt.cm.bwr_r
                            elif globv == 'gpth':
                                clevs = np.arange(-60, 65, 5)
                                cm = plt.cm.PiYG_r
                            cs = m.contourf(plon, plat, anom_comp, clevs, cmap=cm, extend='both')
                        else:
                            cs = m.contourf(plon, plat, anom_comp, extend='both')
                        hatch = m.contourf(plon,plat,mask_pvals,levels=[-1.0, 0.0, 1.0], hatches=["", '.'], alpha=0)

                        plt.title(name2,fontsize=8,fontweight='demibold')

                        # Redraw map
                        m.drawcountries()
                        m.drawcoastlines()


                        cnt += 1


                    else:
                        print 'NO sample FILE AVAILABLE for ' + dset2 + '_' + name2
                        print 'Moving to next model....'
                        cnt += 1

                print "Finalising plot..."
                plt.subplots_adjust(left=0.05,right=0.9,top=0.95,bottom=0.02,wspace=0.1,hspace=0.2)

                # Plot cbar
                axcl = g.add_axes([0.91, 0.15, 0.01, 0.6])
                cbar = plt.colorbar(cs, cax=axcl)
                my.ytickfonts(fontsize=12.)

                # Save
                cstr = ctyp

                if climyr:
                    cstr=cstr+'_35years_'

                if lag:
                    compname = compdir + 'multi_comp_' + cstr + '.' + sample + '.' + type + '.' + globv + \
                               '.' + choosel[l] + '.' + sub + '.from_event' + from_event + '.lag_'+str(edays[lo])+'.png'
                else:
                    compname = compdir + 'multi_comp_'+cstr+'.'+sample+'.' + type + '.' + globv + \
                          '.'+choosel[l]+'.'+sub+'.from_event'+from_event+'.png'
                print 'saving figure as '+compname
                plt.savefig(compname, dpi=150)
                plt.close()