# To plot all CMIP5 models or UM models
# Scatterplots

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

cwd=os.getcwd()
sys.path.append(cwd+'/..')
sys.path.append(cwd+'/../MetBot')
sys.path.append(cwd+'/../RTools')
sys.path.append(cwd+'/../quicks')
import MetBot.dset_dict as dsetdict
import MetBot.mast_dset_dict as mast_dict
import MetBot.dimensions_dict as dim_exdict
import MetBot.mytools as my
import MetBot.MetBlobs as blb
import MetBot.EventStats as stats
import MetBot.mynetcdf as mync
import MetBot.SynopticAnatomy as sy
import scipy

# Which dataset?
whichd='CMIP5' #  if CMIP5 just do CMIP5, if UM do all

if whichd=='UM':
    import dsets_mplot_um as dset_um
    group=False
elif whichd=='CMIP5':
    import dsets_mplot_5group_4plot as dset_mp
    group=True # works for CMIP5 only

### What are we plotting?

# season
seas='DJF'

# x axis - convection
globv1='olr' # olr or omega
levsel1=False
if levsel1:
    choosel1=['500'] # can add a list
else:
    choosel1=['1']
sub_x='scongo'
l=0

# y axis - TTTs
aspect='count' # count or rain
if aspect=='rain':
    under_of='dayof'
    sub_y='alocsub'
    wh_count='event'
    raintype='rainperttt'
        # totrain - gets an ave of rain under each CB and then adds all these aves
        #          - will be highest for those events with lots of rain per TTT and lots of TTTs
        # rainperttt - gets an ave of rain under each CB and then aves them
        #           - will be highest for models with most intense ttt rain regardness of nTTT
        # perc75 - the 75th percentile of mean rain per CB
elif aspect=='count':
    relative = True
    per='year' # TTTs per year
    wh_count = 'blob'  # blob or event
clon1=7.5
clon2=45.0

weightlats=True


# thresholds?
threshtest=True # currently not working for UM - need to check the thresh test intervals for each mod

### Get directories
bkdir=cwd+"/../../CTdata/"
botdir=bkdir+"metbot_multi_dset/"
thisdir=bkdir+"hpaperplot/"
refkey='0'

figdir=thisdir+"scatter_convection_TTT/"
my.mkdir_p(figdir)

### Dsets
dsets='all'
if whichd=='CMIP5':
    dsetnames=['noaa','cmip5']
elif whichd=='UM':
    dsetnames = ['noaa', 'cmip5', 'um']
ndset=len(dsetnames)
ndstr=str(ndset)

### Count total number of models
nm_dset=np.zeros(ndset)
for d in range(ndset):
    dset=dsetnames[d]
    if whichd=='CMIP5':
        nmod=len(dset_mp.dset_deets[dset])
    elif whichd=='UM':
        nmod=len(dsetdict.dset_deets[dset])
    nm_dset[d]=nmod
nallmod=np.sum(nm_dset)
nallmod=int(nallmod)
print nallmod

### Get arrays for output
modnames=[]

### colours
if whichd=='CMIP5':
    if not group:
        cols=['b','g','r','c','m','gold','k',\
            'b','g','r','c','m','gold','k',\
            'b','g','r','c','m','gold','k',\
            'b','g','r','c','m','gold','k']
        markers=["o","o","o","o","o","o","o",\
            "^","^","^","^","^","^","^",\
            "*","*","*","*","*","*","*",\
            "d","d","d","d","d","d","d"]
    elif group:
        grcls = ['fuchsia', 'gold', 'darkblue', 'r', 'blueviolet', 'springgreen']
        grmrs=["o","^","*","d","+","v","h",">"]

    siz = np.full((28), 5)
    siz[0] = 8

elif whichd=='UM':

    cols4noaa = ['b']
    markers4noaa = ["o", "d"]
    siz4noaa = [8]

    cols4um = ['g', 'r', 'c', 'm', 'gold', 'k']
    marker4um = ["^"]
    siz4um = [7]

    cols4had = ['darkorange', 'springgreen']
    marker4had = ["^"]
    siz4had = [7]

    cols4cmip = ['gray']
    marker4cmip = ["*"]
    siz4cmip = [5]


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


### Loop threshs
if threshtest:
    thnames=['lower','actual','upper']
else:
    thnames=['actual']

nthresh=len(thnames)

for t in range(nthresh):

    print "Setting up plot..."
    plt.figure(figsize=[6, 5])
    ax = plt.subplot(111)
    xvals = np.ma.zeros(nallmod, dtype=np.float32)
    yvals = np.ma.zeros(nallmod, dtype=np.float32)
    cnt = 0
    grcnt=np.zeros(7,dtype=np.int8)


    print "Looping datasets"
    for d in range(ndset):
        dset=dsetnames[d]
        dcnt=str(d+1)
        print 'Running on '+dset
        print 'This is dset '+dcnt+' of '+ndstr+' in list'

        if dset != 'cmip5': levc = int(choosel1[l])
        else: levc = int(choosel1[l]) * 100

        ### Models
        mods = 'all'
        if whichd=='CMIP5':
            nmod = len(dset_mp.dset_deets[dset])
            mnames = list(dset_mp.dset_deets[dset])
        elif whichd=='UM':
            nmod = len(dsetdict.dset_deets[dset])
            mnames = list(dsetdict.dset_deets[dset])
        nmstr = str(nmod)
        mdcnt = 0

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
                grmr=grmrs[grcnt[thisgroup-1]]
                grcnt[thisgroup-1]+=1


            ### Convection information for x axis
            # Switch variable if NOAA
            if dset == 'noaa' and globv1 != 'olr':
                if globv1 == 'pr':
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

            vnamedict = globv1 + 'name'
            mastdct = mast_dict.mast_dset_deets[dset2]
            varstr = mastdct[vnamedict]
            dimdict = dim_exdict.dim_deets[globv1][dset2]
            latname = dimdict[1]
            lonname = dimdict[2]
            if dset2 == 'um':
                ys = umdct['climyr']
            else:
                if globv1 != 'omega' and globv1 != 'q' and globv1 != 'gpth':
                    ys = moddct['yrfname']
                else:
                    if name2 == "MIROC5":
                        if globv1 == 'q':
                            ys = moddct['fullrun']
                        elif globv1 == 'omega' or globv1 == 'gpth':
                            ys = '1950_2009'
                        else:
                            print 'variable ' + globv1 + ' has unclear yearname for ' + name2
                    else:
                        ys = moddct['fullrun']

            # Open ltmonmean file
            meanfile = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
                       + name2 + '.' + globv1 + '.mon.mean.' + ys + '.nc'

            print 'Attempting to open ' + meanfile

            if os.path.exists(meanfile):

                if levsel1:
                    ncout = mync.open_multi(meanfile, globv1, name2, \
                                            dataset=dset2, subs=sub_x, levsel=levc)
                else:
                    ncout = mync.open_multi(meanfile, globv1, name2, \
                                            dataset=dset2, subs=sub_x)
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

                # Get regional mean
                if weightlats:
                    latr = np.deg2rad(lat)
                    weights = np.cos(latr)
                    zonmean = np.nanmean(seasmean, axis=1)
                    reg_mean = np.ma.average(zonmean, weights=weights)
                else:
                    reg_mean = np.nanmean(seasmean)

                xvals[cnt]=reg_mean


                ### TTT info for y axis
                ### Get threshold for TTTs
                threshtxt = botdir + 'thresholds.fmin.all_dset.txt'
                print 'Getting threshold for this model'
                print threshtxt
                with open(threshtxt) as f:
                    for line in f:
                        if dset + '\t' + name in line:
                            thresh = line.split()[2]
                            print 'thresh=' + str(thresh)


                thresh = int(thresh)

                if thnames[t]=='actual':
                    thisthresh=thresh
                if thnames[t]=='lower':
                    if dset=='um':
                        thisthresh = thresh - 4
                    else:
                        thisthresh=thresh - 5
                if thnames[t]=='upper':
                    if dset=='um':
                        thisthresh = thresh + 4
                    else:
                        thisthresh=thresh + 5

                thre_str = str(thisthresh)

                # Find TTT data
                outsuf=botdir+dset+"/"+name+"/"+name+"_"

                mbsfile = outsuf + thre_str + '_' + dset + "-olr-0-0.mbs"
                syfile = outsuf + thre_str + '_' + dset + '-OLR.synop'

                ### Open ttt data
                print 'opening metbot files...'

                # using events
                if wh_count=='event':

                    s = sy.SynopticEvents((), [syfile], COL=False)

                    ### Get all events
                    ks = s.events.keys();
                    ks.sort()  # all
                    count_all = len(ks)
                    print "Total CB events =" + str(count_all)
                    key = dset + '-olr-0-' + refkey

                    ### Get season
                    print 'selecting events for this season...'
                    edts = []
                    thesekeys = []
                    for k in ks:
                        e = s.events[k]
                        dts = s.blobs[key]['mbt'][e.ixflags]
                        if len(dts) > 1:
                            dt = dts[len(dts) / 2]
                        else:
                            dt = dts[0]
                        if seas=='DJF' or seas=='NDJFM':
                            if (int(dt[1]) >= mon1) or (int(dt[1]) <= mon2):
                                thesekeys.append(k)
                                edts.append(dt)
                        elif seas=='JF':
                            if (int(dt[1]) >= mon1) and (int(dt[1]) <= mon2):
                                thesekeys.append(k)
                                edts.append(dt)
                        elif seas=='all':
                            thesekeys.append(k)
                            edts.append(dt)
                    edts = np.asarray(edts)
                    yrs = np.unique(edts[:, 0])

                    count_fulldom=len(thesekeys)

                    # Split by domain
                    print 'selecting events for this dom group...'
                    k1, ktmp = stats.spatialsubset(s, thesekeys, cutlon=clon1)
                    k2, k3 = stats.spatialsubset(s, ktmp, cutlon=clon2)
                    count_ttt=len(k2)
                    print "Total CB events in this domain =" + str(count_ttt)

                    keys4rain=k2

                elif wh_count=='blob':
                    refmbs, refmbt, refch = blb.mbopen(mbsfile)
                    count_all=len(refmbt)

                    blob_edts = []
                    blob_edts_regsel =[]
                    for b in range(len(refmbt)):
                        date = refmbt[b]
                        mon = int(date[1])
                        cX = refmbs[b, 3]
                        cY = refmbs[b, 4]

                        if seas == 'all':
                            blob_edts.append(date)

                        elif seas == 'DJF' or seas=='NDJFM':
                            # Check on the month
                            if mon >= mon1 or mon <= mon2:
                                blob_edts.append(date)

                        elif seas == 'JF':
                            # Check on the month
                            if mon >= mon1 and mon <= mon2:
                                blob_edts.append(date)


                        if cX > clon1 and cX < clon2:

                            if seas == 'all':
                                blob_edts_regsel.append(date)

                            elif seas == 'DJF' or seas == 'NDJFM':

                                # Check on the month
                                if mon >= mon1 or mon <= mon2:
                                    blob_edts_regsel.append(date)

                            elif seas == 'JF':
                                # Check on the month
                                if mon >= mon1 and mon <= mon2:
                                    blob_edts_regsel.append(date)


                    blob_edts=np.asarray(blob_edts)
                    yrs = np.unique(blob_edts[:, 0])
                    blob_edts_regsel=np.asarray(blob_edts_regsel)

                    count_fulldom=len(blob_edts)
                    count_ttt=len(blob_edts_regsel)

                if aspect=='count':

                    if per=='year':
                        nys = len(yrs)
                        count_ttt=count_ttt/nys
                        count_fulldom=count_fulldom/nys

                    if relative:
                        count_ttt=float(count_ttt)/float(count_fulldom)*100
                        print "Relative % of TTTs in this domain "+str(count_ttt)

                    yvals[cnt]=count_ttt

                elif aspect=='rain':

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
                    rainfile = botdir + raindset + "/" + rainmod + "." + globp + ".day.mean." + rys + ".nc"
                    print 'Opening ' + rainfile

                    rainout = mync.open_multi(rainfile, globp, rainmod, \
                                              dataset=raindset, subs=sub_y)

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
                    nboxes = float(nlat * nlon)

                    print 'Checking for duplicate timesteps'  # do retain this - IPSL A LR has double tsteps
                    tmp = np.ascontiguousarray(rdtime).view(np.dtype((np.void, rdtime.dtype.itemsize * rdtime.shape[1])))
                    _, idx = np.unique(tmp, return_index=True)
                    rdtime = rdtime[idx]
                    rain = rain[idx, :, :]

                    # Get correct months
                    print 'Selecting the right months'
                    if seas=='DJF' or seas=='NDJFM':
                        raindat2 = np.where((rdtime[:, 1] >= mon1) | (rdtime[:, 1] <= mon2))
                    elif seas=='JF':
                        raindat2 = np.where((rdtime[:, 1] >= mon1) & (rdtime[:, 1] <= mon2))
                    rain = np.squeeze(rain[raindat2, :, :])
                    rdtime = rdtime[raindat2]
                    totdays_hist = len(rdtime)

                    # Get chs
                    print 'Getting dates/chs for historical TTTs'
                    edts = []
                    if under_of == 'under':
                        chs = []
                    ecnt = 1
                    for k in keys4rain:
                        e = s.events[k]
                        dts = s.blobs[key]['mbt'][e.ixflags]
                        for dt in range(len(dts)):
                            if ecnt == 1:
                                edts.append(dts[dt])
                                if under_of == 'under':
                                    chs.append(e.blobs[key]['ch'][e.trk[dt]])
                            else:
                                tmpdt = np.asarray(edts)
                                # Check if it exists already
                                ix = my.ixdtimes(tmpdt, [dts[dt][0]], \
                                                 [dts[dt][1]], [dts[dt][2]], [0])
                                if len(ix) == 0:
                                    edts.append(dts[dt])
                                    if under_of == 'under':
                                        chs.append(e.blobs[key]['ch'][e.trk[dt]])
                            ecnt += 1
                    edts = np.asarray(edts)
                    edts[:, 3] = 0
                    if under_of == 'under':
                        chs = np.asarray(chs)

                    print 'Selecting TTTs from rain data'
                    indices = []
                    if under_of == 'under':
                        chs_4rain = []
                    for edt in range(len(edts)):
                        ix = my.ixdtimes(rdtime, [edts[edt][0]], \
                                         [edts[edt][1]], [edts[edt][2]], [0])
                        if len(ix) >= 1:
                            indices.append(ix)
                            if under_of == 'under':
                                chs_4rain.append(chs[edt])
                    if len(indices) >= 2:
                        indices = np.squeeze(np.asarray(indices))
                        if under_of == 'under':
                            chs_4rain = np.asarray(chs_4rain)
                    else:
                        indices = indices
                        if under_of == 'under':
                            chs_4rain = np.squeeze(np.asarray(chs_4rain))
                    nttt = len(indices)

                    print 'Selecting rain on TTT days'
                    rainsel = rain[indices, :, :]
                    ttt_rain_dates = rdtime[indices]
                    ndt = nttt

                    if under_of == 'under':

                        print 'Selecting rain under TTTs'
                        masked_rain = np.ma.zeros((ndt, nlat, nlon), dtype=np.float32)
                        for rdt in range(ndt):
                            chmask = my.poly2mask(rlon, rlat, chs_4rain[rdt])
                            r = np.ma.MaskedArray(rainsel[rdt, :, :], mask=~chmask)
                            masked_rain[rdt, :, :] = r

                    elif under_of=='dayof':
                        masked_rain=rainsel[:]

                    # Get a timeseries of mean TTT rain from each event
                    print 'Getting a rain value for each TTT event'
                    reg_ttt_sum = np.zeros((len(ttt_rain_dates)), dtype=np.float32)
                    reg_ttt_mean = np.zeros((len(ttt_rain_dates)), dtype=np.float32)
                    for st in range(len(ttt_rain_dates)):
                        reg_ttt_mean[st] = np.ma.mean(masked_rain[st,:,:])

                    # Getting a long term sum or mean
                    tottttrain=np.nansum(reg_ttt_mean)
                    rainperttt=np.nanmean(reg_ttt_mean)
                    per75rain=np.nanpercentile(reg_ttt_mean,75)

                    if raintype=='totrain':
                        yvals[cnt]=tottttrain
                    elif raintype=='rainperttt':
                        yvals[cnt]=rainperttt
                    elif raintype=='perc75':
                        yvals[cnt]=per75rain

                print name
                print xvals[cnt]
                print yvals[cnt]

                if whichd=='CMIP5':

                    if group:
                        colour = grcl
                        mk = grmr
                    else:
                        colour = cols[cnt]
                        mk = markers[cnt]

                    ax.plot(xvals[cnt], yvals[cnt], marker=mk, \
                            color=colour, label=name, markeredgecolor=colour, markersize=siz[cnt], linestyle='None')

                elif whichd=='UM':

                    if dset == 'noaa':
                        ax.plot(xvals[cnt], yvals[cnt], marker=markers4noaa[mdcnt],\
                                color=cols4noaa[0], label=name2, markeredgecolor=cols4noaa[0], markersize=siz4noaa[0],
                                linestyle='None')

                    elif dset == 'cmip5':
                        if name == 'HadGEM2-CC':
                            ax.plot(xvals[cnt], yvals[cnt], marker=marker4had[0], \
                                    color=cols4had[0], label=name2, markeredgecolor=cols4had[0], markersize=siz4had[0],
                                    linestyle='None')
                        elif name == 'HadGEM2-ES':
                            ax.plot(xvals[cnt], yvals[cnt], marker=marker4had[0], \
                                    color=cols4had[1], label=name2, markeredgecolor=cols4had[1], markersize=siz4had[0],
                                    linestyle='None')
                        else:
                            ax.plot(xvals[cnt], yvals[cnt], marker=marker4cmip[0], \
                                    color=cols4cmip[0], label=name2, markeredgecolor=cols4cmip[0],
                                    markersize=siz4cmip[0], linestyle='None')

                    elif dset == 'um':
                        ax.plot(xvals[cnt], yvals[cnt], marker=marker4um[0], \
                                color=cols4um[mdcnt], label=name2, markeredgecolor=cols4um[mdcnt], markersize=siz4um[0],
                                linestyle='None')

            else:
                xvals[cnt] = ma.masked
                yvals[cnt] = ma.masked

            cnt += 1
            mdcnt += 1
            modnames.append(name)

    # Set up plot
    print xvals
    print yvals
    grad, inter, r_value, p_value, std_err = scipy.stats.mstats.linregress(xvals, yvals)
    rsquared = r_value ** 2
    if rsquared > 0.4:
        ax.plot(xvals, (grad * xvals + inter), '-', color='k')


    if sub_x=='subt':
        xlab='Mean subtropical OLR NDJFM'
    elif sub_x=='scongo':
        xlab='Mean OLR in southern Congo DJF'
    else:
        xlab=globv1+'_'+sub_x+'_'+seas
    if aspect=='count':
        if relative:
            ylab = 'Percentage of '+seas+' TTTs between ' + str(clon1) + ' and ' + str(clon2)
        else:
            ylab='Number of TTTs between '+str(clon1)+' and '+str(clon2)
    elif aspect=='rain':
        ylab='TTT rainfall for '+sub_y+': '+raintype

    plt.xlabel(xlab,fontsize=10, fontweight='demibold')
    plt.ylabel(ylab,fontsize=10, fontweight='demibold')

    box=ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='center left',bbox_to_anchor=[1,0.5],fontsize='xx-small',markerscale=0.8,numpoints=1)

    plt.title('$r^2$ '+str(round(rsquared,2)),fontsize=10, loc='right')

    figsuf=""

    if group:
        figsuf=figsuf+'_grouped'
    if aspect=='rain':
        figsuf=figsuf+'_'+under_of+'_'+raintype+'_'+sub_y
    elif aspect == 'count':
        if relative:
            figsuf = figsuf + 'relative'

    if levsel1:
        figsuf = figsuf +'.lev_'+choosel1[l]

    scatterfig=figdir+'/scatter.'+whichd+'.'+globv1+'.'+sub_x+'.seas_'+seas+'.TTT'+aspect+'.'\
               +str(clon1)+'_to_'+str(clon2)+'.'+figsuf+'.countwith_'+wh_count+'.thresh_'+thnames[t]+'.png'
    print 'saving figure as '+scatterfig
    plt.savefig(scatterfig,dpi=150)
    plt.close()