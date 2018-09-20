# Bar chart plotting wrapper
#   to plot a bar chart
#   with colours for each group


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
threshtest=True # Option to run on thresholds + and - 5Wm2 as a test
group=True

weightlats=True

### What are we plotting?

# season
seas='NDJFM'

# # Part b
aspect='count' # count or rain
if aspect=='rain':
    under_of='dayof'
    sub_y='ALL_PR'
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
clon2=55.0


# Part c
# aspect='rain' # count or rain
# if aspect=='rain':
#     under_of='dayof'
#     sub_y='contsub_nh'
#     wh_count='event'
#     raintype='rainperttt'
#         # totrain - gets an ave of rain under each CB and then adds all these aves
#         #          - will be highest for those events with lots of rain per TTT and lots of TTTs
#         # rainperttt - gets an ave of rain under each CB and then aves them
#         #           - will be highest for models with most intense ttt rain regardness of nTTT
#         # perc75 - the 75th percentile of mean rain per CB
# elif aspect=='count':
#     relative = True
#     per='year' # TTTs per year
#     wh_count = 'blob'  # blob or event
# clon1=7.5
# clon2=55.0


### Directories
bkdir=cwd+"/../../CTdata/"
botdir=bkdir+"metbot_multi_dset/"
thisdir=bkdir+"hpaperplot/"
refkey='0'

figdir=thisdir+"barcharts_5group/"
my.mkdir_p(figdir)


### Loop threshs
if threshtest:
    thnames=['lower','actual','upper']
else:
    thnames=['actual']

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
dsetnames=['noaa','cmip5']
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

nthresh=len(thnames)
for t in range(nthresh):

    ### Open arrays for results
    yvals = np.ma.zeros(nallmod, dtype=np.float32)
    cols=["" for x in range(nallmod)]
    modnm=["" for x in range(nallmod)] # creates a list of strings for modnames
    cnt = 0

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

            if thnames[t] == 'actual':
                thisthresh = thresh
            if thnames[t] == 'lower':
                if dset == 'um':
                    thisthresh = thresh - 4
                else:
                    thisthresh = thresh - 5
            if thnames[t] == 'upper':
                if dset == 'um':
                    thisthresh = thresh + 4
                else:
                    thisthresh = thresh + 5

            thre_str = str(thisthresh)

            # Find TTT data
            outsuf = botdir + dset + "/" + name + "/" + name + "_"

            mbsfile = outsuf + thre_str + '_' + dset + "-olr-0-0.mbs"
            syfile = outsuf + thre_str + '_' + dset + '-OLR.synop'

            ### Open ttt data
            print 'opening metbot files...'

            # using events
            if wh_count == 'event':

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
                    if seas == 'DJF' or seas == 'NDJFM':
                        if (int(dt[1]) >= mon1) or (int(dt[1]) <= mon2):
                            thesekeys.append(k)
                            edts.append(dt)
                    elif seas == 'JF':
                        if (int(dt[1]) >= mon1) and (int(dt[1]) <= mon2):
                            thesekeys.append(k)
                            edts.append(dt)
                    elif seas == 'all':
                        thesekeys.append(k)
                        edts.append(dt)
                edts = np.asarray(edts)
                yrs = np.unique(edts[:, 0])

                count_fulldom = len(thesekeys)

                # Split by domain
                print 'selecting events for this dom group...'
                k1, ktmp = stats.spatialsubset(s, thesekeys, cutlon=clon1)
                k2, k3 = stats.spatialsubset(s, ktmp, cutlon=clon2)
                count_ttt = len(k2)
                print "Total CB events in this domain =" + str(count_ttt)

                keys4rain = k2

            elif wh_count == 'blob':
                refmbs, refmbt, refch = blb.mbopen(mbsfile)
                count_all = len(refmbt)

                blob_edts = []
                blob_edts_regsel = []
                for b in range(len(refmbt)):
                    date = refmbt[b]
                    mon = int(date[1])
                    cX = refmbs[b, 3]
                    cY = refmbs[b, 4]

                    if seas == 'all':
                        blob_edts.append(date)

                    elif seas == 'DJF' or seas == 'NDJFM':
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

                blob_edts = np.asarray(blob_edts)
                yrs = np.unique(blob_edts[:, 0])
                blob_edts_regsel = np.asarray(blob_edts_regsel)

                count_fulldom = len(blob_edts)
                count_ttt = len(blob_edts_regsel)

            if aspect == 'count':

                if per == 'year':
                    nys = len(yrs)
                    count_ttt = count_ttt / nys
                    count_fulldom = count_fulldom / nys

                if relative:
                    count_ttt = float(count_ttt) / float(count_fulldom) * 100
                    print "Relative % of TTTs in this domain " + str(count_ttt)

                yvals[cnt] = count_ttt

            elif aspect == 'rain':

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
                if seas == 'DJF' or seas == 'NDJFM':
                    raindat2 = np.where((rdtime[:, 1] >= mon1) | (rdtime[:, 1] <= mon2))
                elif seas == 'JF':
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

                elif under_of == 'dayof':
                    masked_rain = rainsel[:]

                # Get a timeseries of mean TTT rain from each event
                print 'Getting a rain value for each TTT event'
                reg_ttt_sum = np.zeros((len(ttt_rain_dates)), dtype=np.float32)
                reg_ttt_mean = np.zeros((len(ttt_rain_dates)), dtype=np.float32)

                if weightlats:
                    latr=np.deg2rad(rlat)
                    weights=np.cos(latr)


                for st in range(len(ttt_rain_dates)):
                    if weightlats:
                        zonmean_ttt=np.ma.mean(masked_rain[st,:,:],axis=1)
                        regmean_ttt=np.ma.average(zonmean_ttt,weights=weights)
                        reg_ttt_mean[st] = regmean_ttt
                    else:
                        reg_ttt_mean[st] = np.ma.mean(masked_rain[st, :, :])

                # Getting a long term sum or mean
                tottttrain = np.nansum(reg_ttt_mean)
                rainperttt = np.nanmean(reg_ttt_mean)
                per75rain = np.nanpercentile(reg_ttt_mean, 75)

                if raintype == 'totrain':
                    yvals[cnt] = tottttrain
                elif raintype == 'rainperttt':
                    yvals[cnt] = rainperttt
                elif raintype == 'perc75':
                    yvals[cnt] = per75rain

            ### Put name into string list
            if dset=='noaa':
                if aspect=='rain':
                    modnm[cnt] = name+'/'+rainname
                else:
                    modnm[cnt] = name
            else:
                modnm[cnt] = name
            cnt+=1

    figsuf=""
    if group:
        figsuf=figsuf+'_grouped'
    if aspect=='rain':
        figsuf=figsuf+'_'+under_of+'_'+raintype+'_'+sub_y
    elif aspect == 'count':
        if relative:
            figsuf = figsuf + '.relative'


    val=yvals[:]

    # Open text file for results
    file = open(figdir+'txtout.seas_'+seas+'.TTT'+aspect+'.'\
               +str(clon1)+'_to_'+str(clon2)+'.'+figsuf+'.countwith_'+wh_count+'.thresh_'+thnames[t]+'.txt', "w")

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
        tmp=int(val4plot[m])
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
    plt.yticks(pos,modlabels,fontsize=10,fontweight='demibold')
    if aspect=='count':
        plt.xticks(np.arange(0,100,10),fontsize=10, fontweight='demibold')
        plt.xlabel('Percentage of Cloud Bands over the Continent',fontsize=12,fontweight='demibold')
    elif aspect=='rain':
        plt.xticks(np.arange(0,7,1),fontsize=10, fontweight='demibold')
        plt.xlabel('Average Rainfall per Cloud Band',fontsize=12,fontweight='demibold')


    barfig=figdir+'/Barchart.seas_'+seas+'.TTT'+aspect+'.'\
               +str(clon1)+'_to_'+str(clon2)+'.'+figsuf+'.countwith_'+wh_count+'.thresh_'+thnames[t]+'.png'
    plt.savefig(barfig,dpi=150)
    file.close()