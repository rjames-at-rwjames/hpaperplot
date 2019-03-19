# To test variation between model composites
# each model against the other models
# on regmean, with scatterplots to show pvalue and test statistic
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
import MetBot.MetBlobs as blb
import itertools


import scipy
import scipy.interpolate as spi
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


### Running options
which_olr='calc' # 'store' for the one that exists already, 'calc' for new one
test_scr=False
th_offset=False

runs=['opt1']
ctyps=['abs'] #abs is absolute,  anom_mon is rt monthly mean, anom_seas is rt seasonal mean
wcb=['cont'] # which cloud band composite? Options: cont, mada, dbl
spec_col=True
varlist=['olr']
thname='actual'
alphord=True
levsel=False
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


## Interpolation options
interp='none' # 'file' (to import interpolated file) or 'here' to do it in this script
                # or 'none' to not interpolate
if interp=='file':
    fileres='360x180'
    res=1.0
elif interp=='none':
    res=0.0

## Info for options
if domain=='swio':
    sub='SA'

import dsets_paper_28_4plot as dset_mp

if lag:
    edays=[-3,-2,-1,0,1,2,3]
else:
    edays=[0]

cols=['b','g','r','c','m','gold','k',\
    'b','g','r','c','m','gold','k',\
    'b','g','r','c','m','gold','k',\
    'b','g','r','c','m','gold','k']
markers=["o","o","o","o","o","o","o",\
    "^","^","^","^","^","^","^",\
    "*","*","*","*","*","*","*",\
    "d","d","d","d","d","d","d"]

### Get directories
bkdir=cwd+"/../../../CTdata/"
thisdir=bkdir+"/hpaperplot/"
botdir=bkdir+"metbot_multi_dset/"

compdir=thisdir+"comp4paper_test_diffcdr_scatter/"
if lag:
    compdir=thisdir+"comp4paper_test_diffcdr_scatter_lags/"
my.mkdir_p(compdir)

# variable
v=0
globv = varlist[v]
print "Running on "+globv

# levels - if levsel is false this will just be 1 level
l=0
print "Running on "+globv+" at pressure level "+choosel[l]

if seas == 'NDJFM':
    monslist=[1,2,3,11,12]
    mon1 = 11
    mon2 = 3
    nmon = 5
    f_mon = 11
    l_mon = 3


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

    if sample == 'blon':
        best_lon = [33, 58]
        ndays = [50, 50]

        n_cen = [-22, -22]
        s_cen = [-32, -32]

        t_ang = [-60, -50]
        b_ang = [-25, -15]

        f_seas = [11, 11]
        l_seas = [3, 3]

    # Loop abs and anom
    for a in range(len(ctyps)):
        ctyp=ctyps[a]
        print "Running on "+ctyp

        # Loop sample type
        for o in range(len(wcb)):
            type = wcb[o]
            if type == 'cont':
                sind = 0
            elif type == 'mada':
                sind = 1

            # Loop lags
            for lo in range(len(edays)):
                print "Running with a lag of "+str(edays[lo])

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
                        nmod=3
                        nallmod=nmod

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
                                    for zz in range(len(monslist)):
                                        thesemons[zz, :, :] = meandata[monslist[zz] - 1, :, :]
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

                            # if dset == 'cmip5': - not sure why I was doing this - confusion following map script
                            # in map script it was needed because I made a grid with these dimensions
                            #     lat = lat[::-1]
                            #     chosedata = chosedata[:, ::-1, :]

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

                            thdiff=243.0-float(thresh)

                            outsuf = botpath + name + '_'
                            syfile = outsuf + thre_str + '_' + dset + '-OLR.synop'
                            print 'opening metbot files...'
                            mbsfile = outsuf + thre_str + '_' + dset + "-olr-0-0.mbs"
                            refmbs, refmbt, refch = blb.mbopen(mbsfile)
                            refmbt[:, 3] = 0
                            s = sy.SynopticEvents((), [syfile], COL=False)
                            ks = s.events.keys();ks.sort()  # all
                            refkey = s.mbskeys[0]

                            # Select the first day
                            if from_event == 'first':
                                ev_dts = []
                                ev_keys = []
                                ev_cXs = []
                                ev_chs = []

                                for k in ks:
                                    e = s.events[k]
                                    dts = s.blobs[refkey]['mbt'][e.ixflags]
                                    for dt in range(len(dts)):
                                        x, y = e.trkcX[dt], e.trkcY[dt]
                                        ev_dts.append(dts[dt])
                                        ev_keys.append(k)
                                        ev_cXs.append(x)
                                        ev_chs.append(e.blobs[refkey]['ch'][e.trk[dt]])

                                ev_dts = np.asarray(ev_dts)
                                ev_dts[:, 3] = 0
                                ev_keys = np.asarray(ev_keys)
                                ev_cXs = np.asarray(ev_cXs)
                                ev_chs = np.asarray(ev_chs)

                            ### Get array of centroids and angles and chs
                            edts = []
                            cXs = []
                            cYs = []
                            degs = []
                            mons = []
                            chs = []
                            ekeys = []
                            olrmns = []


                            for b in range(len(refmbt)):
                                date = refmbt[b]
                                mon = int(date[1])
                                cX = refmbs[b, 3]
                                cY = refmbs[b, 4]
                                deg = refmbs[b, 2]
                                olrmn = refmbs[b, 11]

                                if from_event == 'first':
                                    # print 'Checking if the date is the first day of an event'
                                    ix = my.ixdtimes(ev_dts, [date[0]], \
                                                     [date[1]], [date[2]], [0])
                                    if len(ix) == 1:
                                        key = ev_keys[ix]
                                        e = s.events[key[0]]
                                        dts = s.blobs[refkey]['mbt'][e.ixflags]
                                        if dts[0, 0] == date[0]:
                                            if dts[0, 1] == date[1]:
                                                if dts[0, 2] == date[2]:
                                                    #               print 'it is the first date, so we keep it'
                                                    edts.append(date)
                                                    cXs.append(cX)
                                                    cYs.append(cY)
                                                    degs.append(deg)
                                                    mons.append(mon)
                                                    chs.append(e.blobs[refkey]['ch'][e.trk[0]])
                                                    ekeys.append(key)
                                                    olrmns.append(olrmn)
                                            #     else:
                                            #         print 'this is not the first day... ignore'
                                            # else:
                                            #     print 'this is not the first day... ignore'
                                    elif len(ix) > 1:
                                        #   print 'there is more than one event on this day'
                                        #   print 'lets find the centroid that matches'
                                        todays_cXs = ev_cXs[ix]
                                        index2 = np.where(todays_cXs == cX)[0]
                                        if len(index2) != 1:
                                            print 'Error - centroid not matching'
                                        index3 = ix[index2]
                                        key = ev_keys[index3]
                                        #   print 'selecting event with matching centroid'
                                        e = s.events[key[0]]
                                        dts = s.blobs[refkey]['mbt'][e.ixflags]
                                        #   print 'but is it the first date?'
                                        if dts[0, 0] == date[0]:
                                            if dts[0, 1] == date[1]:
                                                if dts[0, 2] == date[2]:
                                                    #               print 'it is the first date, so we keep it'
                                                    edts.append(date)
                                                    cXs.append(cX)
                                                    cYs.append(cY)
                                                    degs.append(deg)
                                                    mons.append(mon)
                                                    chs.append(e.blobs[refkey]['ch'][e.trk[0]])
                                                    ekeys.append(key)
                                                    olrmns.append(olrmn)
                                            #     else:
                                            #         print 'this is not the first day... ignore'
                                            # else:
                                            #     print 'this is not the first day... ignore'

                            edts = np.asarray(edts)
                            edts[:, 3] = 0
                            cXs = np.asarray(cXs)
                            cYs = np.asarray(cYs)
                            degs = np.asarray(degs)
                            mons = np.asarray(mons)
                            chs = np.asarray(chs)
                            ekeys = np.asarray(ekeys)
                            olrmns = np.asarray(olrmns)

                            # ### Limit to certain mons - removed because this is not helpful - creates difference to original sample procedure
                            # if seas == 'NDJFM':  # because the > and < statement below only works for summer
                            #     pick1 = np.where(mons >= f_mon)[0]
                            #     pick2 = np.where(mons <= l_mon)[0]
                            #     pick = np.concatenate([pick1, pick2])
                            #     edts = edts[pick]
                            #     cXs = cXs[pick]
                            #     cYs = cYs[pick]
                            #     degs = degs[pick]
                            #     mons = mons[pick]
                            #     chs = chs[pick]
                            #     ekeys = ekeys[pick]
                            #     olrmns = olrmns[pick]

                            ### Find sample
                            if sample == 'blon' or sample == 'blon2':
                                tmp_edts = []
                                tmp_cXs = []
                                tmp_cYs = []
                                tmp_degs = []
                                tmp_mons = []
                                tmp_chs = []
                                tmp_ekeys = []
                                tmp_olrmns = []

                                for b in range(len(edts)):
                                    date = edts[b]
                                    mon = mons[b]
                                    cX = cXs[b]
                                    cY = cYs[b]
                                    deg = degs[b]
                                    ch = chs[b]
                                    thiskey = ekeys[b]
                                    thisolr = olrmns[b]

                                    # Check on the latitude of centroid
                                    if cY > s_cen[sind] and cY < n_cen[sind]:

                                        # Check on the angle
                                        if deg > t_ang[sind] and deg < b_ang[sind]:

                                            # Check on the month
                                            if mon >= f_seas[sind] or mon <= l_seas[sind]:
                                                tmp_edts.append(date)
                                                tmp_cXs.append(cX)
                                                tmp_cYs.append(cY)
                                                tmp_degs.append(deg)
                                                tmp_mons.append(mon)
                                                tmp_chs.append(ch)
                                                tmp_ekeys.append(thiskey)
                                                tmp_olrmns.append(thisolr)

                                tmp_edts = np.asarray(tmp_edts)
                                tmp_cXs = np.asarray(tmp_cXs)
                                tmp_cYs = np.asarray(tmp_cYs)
                                tmp_degs = np.asarray(tmp_degs)
                                tmp_mons = np.asarray(tmp_mons)
                                tmp_chs = np.asarray(tmp_chs)
                                tmp_ekeys = np.asarray(tmp_ekeys)
                                tmp_olrmns = np.asarray(tmp_olrmns)
                                print 'Shortlist of ' + str(len(tmp_edts)) + ' continental cloudbands'

                                # Get the 50 closest to best_lon
                                dists = best_lon[sind] - tmp_cXs
                                abs_dists = np.absolute(dists)
                                inds = np.argsort(abs_dists)
                                dists_sort = abs_dists[inds]
                                first50_ind = inds[0:ndays[0]]

                                smp_cXs = tmp_cXs[first50_ind]
                                smp_cYs = tmp_cYs[first50_ind]
                                smp_edts = tmp_edts[first50_ind]
                                smp_degs = tmp_degs[first50_ind]
                                smp_mons = tmp_mons[first50_ind]
                                smp_chs = tmp_chs[first50_ind]
                                smp_olrmns = tmp_olrmns[first50_ind]
                                smp_ekeys = tmp_ekeys[first50_ind]
                                smp_ekeys = smp_ekeys[:,0]

                                print 'Sampled ' + str(len(smp_cXs)) + ' cloudbands'
                                if len(smp_cXs) < 50:
                                    print 'LESS THAN 50 CLOUDBANDS SAMPLED'
                                    tag = ' ONLY_' + str(len(smp_cXs))
                                else:
                                    tag = ''

                                # Put them in order
                                keyinds = np.argsort(smp_ekeys)
                                smp_cXs = smp_cXs[keyinds]
                                smp_cYs = smp_cYs[keyinds]
                                smp_edts = smp_edts[keyinds]
                                smp_degs = smp_degs[keyinds]
                                smp_mons = smp_mons[keyinds]
                                smp_chs = smp_chs[keyinds]
                                smp_ekeys = smp_ekeys[keyinds]
                                smp_olrmns = smp_olrmns[keyinds]

                            print 'Printing data for OLR means stored in mbs files'
                            print smp_olrmns


                            #   then use the outline to cut out data from sample
                            masked_var=np.ma.zeros((nsamp,this_nlat,this_nlon),dtype=np.float32)
                            cbmeans=np.zeros(nsamp,dtype=np.float32)
                            for rdt in range(nsamp):
                                chmask = my.poly2mask(lon, lat, smp_chs[rdt])
                                r = np.ma.MaskedArray(chosedata[rdt, :, :], mask=~chmask)
                                masked_var[rdt, :, :] = r
                                cbmeans[rdt]= np.ma.mean(r)

                            print 'Printing data for OLR means calculated here'
                            print cbmeans

                            if which_olr=='calc':
                                new_cbmeans=cbmeans
                            elif which_olr=='store':
                                new_cbmeans=smp_olrmns

                            # Adjust if thresh adjustment
                            if th_offset:
                                new_cbmeans=new_cbmeans+thdiff

                            collect[cnt-1]=new_cbmeans
                            print collect[cnt-1]

                            cnt +=1
                            modnames.append(name2)

                        else:
                            print 'NO sample FILE AVAILABLE for ' + dset2 + '_' + name2
                            print 'Moving to next model....'
                            cnt += 1
                            modnames.append(name2)


                # Loop models to compare with NOAA
                loopstats_u=np.zeros(nallmod-1,dtype=np.float32)
                loopps_u=np.zeros(nallmod-1,dtype=np.float32)

                loopstats_t=np.zeros(nallmod-1,dtype=np.float32)
                loopps_t=np.zeros(nallmod-1,dtype=np.float32)

                keynames=[]

                # Set up plot - mann whitney
                print "Setting up plot..."
                plt.figure(figsize=[6, 5])
                ax = plt.subplot(111)

                for co in range(nallmod-1):
                    m1=0
                    m2=co+1

                    mname1=modnames[m1]
                    mname2=modnames[m2]
                    print 'Comparing '+mname1
                    print 'and '+mname2


                    vals1=collect[m1]
                    vals2=collect[m2]

                    ustat, upval = scipy.stats.mannwhitneyu(vals1, vals2, alternative='two-sided')

                    tstat, tpval = scipy.stats.ttest_ind(vals1, vals2)

                    loopstats_u[co]=ustat
                    loopps_u[co]=upval

                    loopstats_t[co]=abs(tstat)
                    loopps_t[co]=tpval

                    thisname=mname1+'_'+mname2

                    keynames.append(thisname)

                    ax.plot(loopstats_t[co],loopps_t[co],marker=markers[co], \
                        color=cols[co], label=thisname, markeredgecolor=cols[co], markersize=5, linestyle='None')

                # Finalising plot
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(loc='center left', bbox_to_anchor=[1, 0.5], fontsize='xx-small', markerscale=0.8, numpoints=1)

                # Saving output plot
                if test_scr:
                    mods='testmodels'
                else:
                    mods='allmod'
                bit=""
                if th_offset:
                    bit=bit+"th_offset"

                compname= compdir + 'statsplot_test.t_test.'+globv+'.'+choosel[l]+'.'+ctyp+'.interp_'+interp+'_'+str(res)+'.models_'+mods+'.'+bit+'.input_'+which_olr+'.png'
                plt.savefig(compname, dpi=150)
                plt.close()