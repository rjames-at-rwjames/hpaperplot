# To test variation between model composites
# using ANOVA and Kruskal-Wallis (non-parametric)
# on regmean, with boxplots to show range
# regmean is based on CB footprint

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
import MetBot.SynopticAnatomy as sy

import scipy
import scipy.interpolate as spi
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


### Running options
test_scr=False

runs=['opt1']
ctyps=['anom_seas'] #abs is absolute,  anom_mon is rt monthly mean, anom_seas is rt seasonal mean
wcb=['cont'] # which cloud band composite? Options: cont, mada, dbl
spec_col=True
varlist=['q']
thname='actual'
alphord=True
levsel=True
if levsel:
    choosel=['850'] # can add a list
else:
    choosel=['1']
domain='swio'
lag=False
seas='NDJFM'
climyr='spec' # this is to use new climatology files which are based on only 35 years
                # 'spec' is 35 years
                # 'prev' is previous files - all different climatologies
refkey='0'              # for MetBot


## Interpolation options
interp='file' # 'file' (to import interpolated file) or 'here' to do it in this script
if interp=='file':
    fileres='360x180'
    res=1.0

## Info for options
if domain=='swio':
    sub='SA'

import dsets_paper_28_4plot as dset_mp

if lag:
    edays=[-3,-2,-1,0,1,2,3]
else:
    edays=[0]


### Get directories
bkdir=cwd+"/../../../CTdata/"
thisdir=bkdir+"/hpaperplot/"
botdir=bkdir+"metbot_multi_dset/"

compdir=thisdir+"comp4paper_anova_reg/"
if lag:
    compdir=thisdir+"comp4paper_anova_reg_lags/"
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

                # Set up plot
                print "Setting up plot..."
                plt.figure(figsize=[10, 5])

                cnt = 1
                modnames = []

                ### Dsets
                if test_scr:
                    dsets='spec'
                    ndset=1
                    dsetnames=['cmip5']
                else:
                    dsets='all'
                    ndset=len(dset_mp.dset_deets)
                    dsetnames=['noaa','cmip5']
                ndstr=str(ndset)

                ### Count total number of models
                nm_dset = np.zeros(ndset)
                for d in range(ndset):
                    dset = dsetnames[d]
                    nmod = len(dset_mp.dset_deets[dset])
                    nm_dset[d] = nmod
                nallmod = np.sum(nm_dset)
                nallmod = int(nallmod)
                print nallmod

                # Make array for models and composites
                collect = [np.zeros(50, dtype=np.float32)] * nallmod

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
                        botpath=bkdir + 'metbot_multi_dset/' + dset + '/' + name + '/'
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
                                    if name2=='cdr':
                                        ysclim='1979_2013'
                                    else:
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

                                print 'Opening ' + meanfile

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


                                else:
                                    print 'NO MEAN FILE AVAILABLE for ' + dset2 + '_' + name2
                                    print 'Setting mean to zero'
                                    meandata=np.zeros((12,nlat,nlon), dtype=np.float32)


                            if ctyp=='abs':
                                chosedata=smpdata
                            elif ctyp=='anom_seas':
                                chosedata=anoms

                            if dset == 'cmip5':
                                lat = lat[::-1]
                                chosedata = chosedata[:, ::-1, :]

                            # Get the cloudband outlines for sample
                            #   first open synop file
                            #   first get threshold
                            print 'getting threshold....'
                            threshtxt = botdir + 'thresholds.fmin.all_dset.txt'
                            with open(threshtxt) as f:
                                for line in f:
                                    if dset + '\t' + name in line:
                                        thresh = line.split()[2]
                                        print 'thresh=' + str(thresh)
                            thresh = int(thresh)
                            thre_str = str(thresh)

                            outsuf = botpath + name + '_'
                            syfile = outsuf + thre_str + '_' + dset + '-OLR.synop'
                            print 'opening metbot files...'
                            s = sy.SynopticEvents((), [syfile], COL=False)
                            key = dset + '-olr-0-' + refkey
                            ks = s.events.keys();ks.sort()  # all

                            #   then get list of dates and outlines for these events
                            edts = []
                            chs = []
                            ecnt = 1
                            for k in ks:
                                e = s.events[k]
                                dts = s.blobs[key]['mbt'][e.ixflags]
                                for dt in range(len(dts)):
                                    if ecnt == 1:
                                        edts.append(dts[dt])
                                        chs.append(e.blobs[key]['ch'][e.trk[dt]])
                                    else:
                                        tmpdt = np.asarray(edts)
                                        # Check if it exists already
                                        ix = my.ixdtimes(tmpdt, [dts[dt][0]], \
                                                         [dts[dt][1]], [dts[dt][2]], [dts[dt][3]] )
                                        if len(ix) == 0:
                                            edts.append(dts[dt])
                                            chs.append(e.blobs[key]['ch'][e.trk[dt]])
                                    ecnt +=1
                            edts = np.asarray(edts)
                            edts[:, 3] = 0
                            chs = np.asarray(chs)

                            #   then select the outlines for the sample
                            date_check = []
                            chs_smpl = []
                            for edt in range(len(edts)):
                                ix = my.ixdtimes(smpdtime, [edts[edt][0]], \
                                                 [edts[edt][1]], [edts[edt][2]], [0])
                                if len(ix) >= 1:
                                    date_check.append(edts[edt])
                                    chs_smpl.append(chs[edt])
                            date_check = np.squeeze(np.asarray(date_check))
                            chs_smpl = np.squeeze(np.asarray(chs_smpl))

                            #   then use the outline to cut out data from sample
                            masked_var=np.ma.zeros((nsamp,this_nlat,this_nlon),dtype=np.float32)
                            cbmeans=np.zeros(nsamp,dtype=np.float32)
                            for rdt in range(nsamp):
                                chmask = my.poly2mask(lon, lat, chs_smpl[rdt])
                                r = np.ma.MaskedArray(chosedata[rdt, :, :], mask=~chmask)
                                masked_var[rdt, :, :] = r
                                cbmeans[rdt]= np.ma.mean(r)

                            # Collect this model's data for anova
                            collect[cnt-1]=cbmeans
                            print collect[cnt-1]

                            # Make boxplot
                            plt.boxplot(cbmeans,positions=[cnt],notch=0,sym='+',vert=1,whis=1.5)

                            cnt +=1
                            modnames.append(name2)

                        else:
                            print 'NO sample FILE AVAILABLE for ' + dset2 + '_' + name2
                            print 'Moving to next model....'
                            cnt += 1
                            modnames.append(name2)

                # anova test
                print "Sample data collected:"
                print collect
                print "Running ANOVA..."
                f, p = scipy.stats.f_oneway(*collect)
                print "F stat is:"
                print f
                print "p value is:"
                print p

                "Running Kruskal-Wallis..."
                K, pval = scipy.stats.kruskal(*collect)
                print "Output stat is:"
                print K
                print "p value is:"
                print pval

                # Finalise plot
                print "Finalising plot..."
                xposs = np.arange(1, cnt)
                plt.xticks(xposs, modnames, rotation='vertical', fontsize='xx-small')
                print 'Plotting xticks for modnames '
                print modnames
                print 'at positions ' + str(xposs)
                plt.xlim(0, cnt)
                print 'Plotting xlims from 0 to ' + str(cnt)
                plt.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.95)
                plt.title('Range of values for '+globv+'   ANOVA='+str(round(f,1))+' with p='+str(round(p,2))+'   K-W='+str(round(K,1))+' with p='+str(round(pval,2)))

                # Saving output plot
                if test_scr:
                    mods='testmodels'
                else:
                    mods='allmod'

                compname= compdir + 'statsplot_test.'+globv+'.'+choosel[l]+'.'+ctyp+'.interp_'+interp+'_'+str(res)+'.models_'+mods+'.png'
                plt.savefig(compname, dpi=150)
                plt.close()