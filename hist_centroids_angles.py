# Wrapper to plot distribution of centroids and angles from different models
#   this one plots the UM models in a grid


import os
import sys

import matplotlib.pyplot as plt
import numpy as np

cwd=os.getcwd()
sys.path.append(cwd+'/..')
sys.path.append(cwd+'/../MetBot')
sys.path.append(cwd+'/../RTools')
import MetBot.SynopticAnatomy as sy
import MetBot.MetBlobs as blb
import MetBot.mytools as my
import dsets_mplot_5group_4plot as dset_mp

### Running options
cenlonplot=False
cenlatplot=False
angleplot=False
monplot=False

scatter_lon_angle=True
scatter_lat_angle=False
scatter_lon_lat=False

scatter_lon_mon=False
scatter_lat_mon=False
scatter_ang_mon=False

one_seas=True
seas_pick='NDJFM'
from_event='first' # first or all (all just uses refmbt)

show_sample=True
show_cont_sample=True
show_mada_sample=True
sample='blon2'

group=False
alphord=True


title=True
xplots = 4
yplots = 7
totsize=[9, 12]

testyear=False  # plot based on 1 year of test data
testfile=False
threshtest=True # Option to run on thresholds + and - 5Wm2 as a test
monthstr = ['Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']

if group:
    grcls=['fuchsia','darkorange','b','r','blueviolet','springgreen']


### Directory
bkdir=cwd+"/../../CTdata/"
botdir=bkdir+"metbot_multi_dset/"
thisdir=bkdir+"/hpaperplot/"

figdir=thisdir+"allCMIP_cen_ang_figs/"
my.mkdir_p(figdir)

if seas_pick=='NDJFM':

    f_mon = 11
    l_mon = 3

# Get sample deets - first element cont, second element mada
if sample=='blon':

    best_lon = [33,58]
    ndays=[50,50]

    n_cen = [-22,-22]
    s_cen = [-32,-32]

    t_ang = [-60,-50]
    b_ang = [-25,-15]

    f_seas = [11,11]
    l_seas = [3,3]

if sample == 'blon2':

    best_lon = [33,60]
    ndays=[50,50]

    n_cen = [-26,-24]
    s_cen = [-30,-28]

    t_ang = [-50,-40]
    b_ang = [-35,-25]

    f_seas = [11,11]
    l_seas = [3,3]


if sample=='blat':

    best_lat = [-28,-26]
    ndays=[50,50]

    w_cen = [28,55]
    e_cen = [38,65]

    t_ang = [-60,-50]
    b_ang = [-25,-15]

    f_seas = [11,11]
    l_seas = [3,3]

if sample=='bang':

    best_deg = [-42,-32]
    ndays=[50,50]

    w_cen = [25,52]
    e_cen = [40,67]

    n_cen = [-22,-22]
    s_cen = [-32,-32]

    f_seas = [11,11]
    l_seas = [3,3]

### Loop threshs
if threshtest:
    thnames=['lower','actual','upper']
else:
    thnames=['actual']

nthresh=len(thnames)
for t in range(nthresh):

    # Set up plot
    print "Setting up plot..."
    ### Open figures
    if cenlonplot: plt.figure(num='cenlon',figsize=totsize)
    if cenlatplot: plt.figure(num='cenlat',figsize=totsize)
    if angleplot: plt.figure(num='angle',figsize=totsize)
    if monplot: plt.figure(num='month',figsize=totsize)

    if scatter_lon_angle: plt.figure(num='lon_ang',figsize=totsize)
    if scatter_lat_angle: plt.figure(num='lat_ang', figsize=totsize)
    if scatter_lon_lat: plt.figure(num='lon_lat', figsize=totsize)

    if scatter_lon_mon: plt.figure(num='lon_mon', figsize=totsize)
    if scatter_lat_mon: plt.figure(num='lat_mon', figsize=totsize)
    if scatter_ang_mon: plt.figure(num='ang_mon', figsize=totsize)

    cnt = 1

    ### Loop datasets
    dsets='all'
    ndset = len(dset_mp.dset_deets)
    # dsetnames=list(dset_mp.dset_deets)
    dsetnames = ['noaa', 'cmip5']
    ndstr = str(ndset)

    print "Looping datasets"
    for d in range(ndset):
        dset = dsetnames[d]
        dcnt = str(d + 1)
        print 'Running on ' + dset
        print 'This is dset ' + dcnt + ' of ' + ndstr + ' in list'

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

            elif alphord:
                mnames = sorted(mnames_tmp, key=lambda s: s.lower())
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

            ### Find location of mbs file
            sydir=botdir+dset+"/"+name+"/"
            if testyear: outdir=sydir+'test/'
            else: outdir=sydir
            outsuf=outdir+name+'_'

            ### Get thresh
            if testyear:
                threshtxt = botdir + 'thresholds.fmin.' + dset + '.test.txt'
            else:
                threshtxt = botdir + 'thresholds.fmin.all_dset.txt'
            with open(threshtxt) as f:
                for line in f:
                    if dset+'\t'+name in line:
                        thresh = line.split()[2]
                        print 'thresh='+str(thresh)

            thresh = int(thresh)

            if thnames[t]=='actual':
                thisthresh=thresh
            if thnames[t]=='lower':
                thisthresh=thresh - 5
            if thnames[t]=='upper':
                thisthresh=thresh + 5

            thre_str = str(int(thisthresh))

            ###  Open mbs file
            mbsfile = outsuf + thre_str + '_' + dset + "-olr-0-0.mbs"
            refmbs, refmbt, refch = blb.mbopen(mbsfile)
            refmbt[:, 3] = 0

            ### Count number of TTCB days
            count_all = len(refmbt)
            print "Total CBs flagged =" + str(count_all)

            # Count number of events
            if from_event=='first':
                syfile = outsuf + thre_str + '_' + dset + '-OLR.synop'
                s = sy.SynopticEvents((), [syfile], COL=False)
                refkey = s.mbskeys[0]
                ks = s.events.keys();ks.sort()
                count_events = str(int(len(ks)))
                print "Total CB events =" + str(count_events)

                ev_dts=[]
                ev_keys=[]
                ev_cXs=[]

                for k in ks:
                    e = s.events[k]
                    dts = s.blobs[refkey]['mbt'][e.ixflags]
                    for dt in range(len(dts)):
                        x,y = e.trkcX[dt], e.trkcY[dt]
                        ev_dts.append(dts[dt])
                        ev_keys.append(k)
                        ev_cXs.append(x)

                ev_dts=np.asarray(ev_dts)
                ev_dts[:,3]=0
                ev_keys=np.asarray(ev_keys)
                ev_cXs=np.asarray(ev_cXs)

            ### Get array of centroids and angles
            edts = []
            cXs = []
            cYs = []
            degs = []
            mons = []

            for b in range(len(refmbt)):
                date = refmbt[b]
                mon = int(date[1])
                cX = refmbs[b, 3]
                cY = refmbs[b, 4]
                deg = refmbs[b, 2]

                if from_event=='all':
                    edts.append(date)
                    cXs.append(cX)
                    cYs.append(cY)
                    degs.append(deg)
                    mons.append(mon)
                elif from_event=='first':
                    #print 'Checking if the date is the first day of an event'
                    ix = my.ixdtimes(ev_dts, [date[0]], \
                                     [date[1]], [date[2]], [0])
                    if len(ix)==1:
                        key=ev_keys[ix]
                        e = s.events[key[0]]
                        dts = s.blobs[refkey]['mbt'][e.ixflags]
                        if dts[0,0]==date[0]:
                            if dts[0,1]==date[1]:
                                if dts[0,2]==date[2]:
                     #               print 'it is the first date, so we keep it'
                                    edts.append(date)
                                    cXs.append(cX)
                                    cYs.append(cY)
                                    degs.append(deg)
                                    mons.append(mon)
                            #     else:
                            #         print 'this is not the first day... ignore'
                            # else:
                            #     print 'this is not the first day... ignore'
                    elif len(ix)>1:
                     #   print 'there is more than one event on this day'
                     #   print 'lets find the centroid that matches'
                        todays_cXs=ev_cXs[ix]
                        index2=np.where(todays_cXs==cX)[0]
                        if len(index2)!=1:
                            print 'Error - centroid not matching'
                        index3=ix[index2]
                        key=ev_keys[index3]
                     #   print 'selecting event with matching centroid'
                        e = s.events[key[0]]
                        dts = s.blobs[refkey]['mbt'][e.ixflags]
                     #   print 'but is it the first date?'
                        if dts[0,0]==date[0]:
                            if dts[0,1]==date[1]:
                                if dts[0,2]==date[2]:
                     #               print 'it is the first date, so we keep it'
                                    edts.append(date)
                                    cXs.append(cX)
                                    cYs.append(cY)
                                    degs.append(deg)
                                    mons.append(mon)
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

            ### Limit to certain mons
            if one_seas:
                if seas_pick=='NDJFM': # because the > and < statement below only works for summer
                    pick1=np.where(mons >= f_mon)[0]
                    pick2=np.where(mons <= l_mon)[0]
                    pick=np.concatenate([pick1,pick2])
                    edts=edts[pick]
                    cXs=cXs[pick]
                    cYs=cYs[pick]
                    degs=degs[pick]
                    mons=mons[pick]

            ### Find sample
            if show_sample:
                if sample=='blon' or sample=='blon2':
                    if show_cont_sample:
                        tmp_edts=[]
                        tmp_cXs=[]
                        tmp_cYs=[]
                        tmp_degs=[]
                        tmp_mons=[]

                        for b in range(len(edts)):
                            date=edts[b]
                            mon=mons[b]
                            cX=cXs[b]
                            cY=cYs[b]
                            deg=degs[b]

                            # Check on the latitude of centroid
                            if cY > s_cen[0] and cY < n_cen[0]:

                                # Check on the angle
                                if deg > t_ang[0] and deg < b_ang[0]:

                                    # Check on the month
                                    if mon >= f_seas[0] or mon <= l_seas[0]:

                                        tmp_edts.append(date)
                                        tmp_cXs.append(cX)
                                        tmp_cYs.append(cY)
                                        tmp_degs.append(deg)
                                        tmp_mons.append(mon)

                        tmp_edts=np.asarray(tmp_edts)
                        tmp_cXs=np.asarray(tmp_cXs)
                        tmp_cYs=np.asarray(tmp_cYs)
                        tmp_degs=np.asarray(tmp_degs)
                        tmp_mons=np.asarray(tmp_mons)
                        print 'Shortlist of '+str(len(tmp_edts))+' continental cloudbands'

                        # Get the 50 closest to best_lon
                        dists = best_lon[0] - tmp_cXs
                        abs_dists = np.absolute(dists)
                        inds = np.argsort(abs_dists)
                        dists_sort = abs_dists[inds]
                        first50_ind = inds[0:ndays[0]]

                        cont_cXs=tmp_cXs[first50_ind]
                        cont_cYs=tmp_cYs[first50_ind]
                        cont_edts=tmp_edts[first50_ind]
                        cont_degs=tmp_degs[first50_ind]
                        cont_mons=tmp_mons[first50_ind]
                        print 'Sampled '+str(len(cont_cXs))+' continental cloudbands'
                        if len(cont_cXs)<50:
                            print 'LESS THAN 50 CLOUDBANDS SAMPLED'
                            tag = ' ONLY_'+str(len(cont_cXs))
                        else:
                            tag = ''


                    if show_mada_sample:
                        tmp_edts = []
                        tmp_cXs = []
                        tmp_cYs = []
                        tmp_degs = []
                        tmp_mons = []

                        for b in range(len(edts)):
                            date = edts[b]
                            mon = mons[b]
                            cX = cXs[b]
                            cY = cYs[b]
                            deg = degs[b]

                            # Check on the latitude of centroid
                            if cY > s_cen[1] and cY < n_cen[1]:

                                # Check on the angle
                                if deg > t_ang[1] and deg < b_ang[1]:

                                    # Check on the month
                                    if mon >= f_seas[1] or mon <= l_seas[1]:
                                        tmp_edts.append(date)
                                        tmp_cXs.append(cX)
                                        tmp_cYs.append(cY)
                                        tmp_degs.append(deg)
                                        tmp_mons.append(mon)

                        tmp_edts = np.asarray(tmp_edts)
                        tmp_cXs = np.asarray(tmp_cXs)
                        tmp_cYs = np.asarray(tmp_cYs)
                        tmp_degs = np.asarray(tmp_degs)
                        tmp_mons = np.asarray(tmp_mons)
                        print 'Shortlist of '+str(len(tmp_edts))+' madagascan cloudbands'

                        # Get the 50 closest to best_lon
                        dists = best_lon[1] - tmp_cXs
                        abs_dists = np.absolute(dists)
                        inds = np.argsort(abs_dists)
                        dists_sort = abs_dists[inds]
                        first50_ind = inds[0:ndays[1]]

                        mada_cXs = tmp_cXs[first50_ind]
                        mada_cYs = tmp_cYs[first50_ind]
                        mada_edts = tmp_edts[first50_ind]
                        mada_degs = tmp_degs[first50_ind]
                        mada_mons = tmp_mons[first50_ind]
                        print 'Sampled '+str(len(mada_cXs))+' madagascan cloudbands'
                        if len(mada_cXs)<50:
                            print 'LESS THAN 50 CLOUDBANDS SAMPLED'
                            tag = ' ONLY_' + str(len(mada_cXs))
                        else:
                            tag = ''

                if sample == 'blat':
                    if show_cont_sample:
                        tmp_edts = []
                        tmp_cXs = []
                        tmp_cYs = []
                        tmp_degs = []
                        tmp_mons = []

                        for b in range(len(edts)):
                            date = edts[b]
                            mon = mons[b]
                            cX = cXs[b]
                            cY = cYs[b]
                            deg = degs[b]

                            # Check on the longitude of centroid
                            if cX > w_cen[0] and cX < e_cen[0]:

                                # Check on the angle
                                if deg > t_ang[0] and deg < b_ang[0]:

                                    # Check on the month
                                    if mon >= f_seas[0] or mon <= l_seas[0]:
                                        tmp_edts.append(date)
                                        tmp_cXs.append(cX)
                                        tmp_cYs.append(cY)
                                        tmp_degs.append(deg)
                                        tmp_mons.append(mon)

                        tmp_edts = np.asarray(tmp_edts)
                        tmp_cXs = np.asarray(tmp_cXs)
                        tmp_cYs = np.asarray(tmp_cYs)
                        tmp_degs = np.asarray(tmp_degs)
                        tmp_mons = np.asarray(tmp_mons)
                        print 'Shortlist of '+str(len(tmp_edts))+' continental cloudbands'

                        # Get the 50 closest to best_lon
                        dists = best_lat[0] - tmp_cYs
                        abs_dists = np.absolute(dists)
                        inds = np.argsort(abs_dists)
                        dists_sort = abs_dists[inds]
                        first50_ind = inds[0:ndays[0]]

                        cont_cXs = tmp_cXs[first50_ind]
                        cont_cYs = tmp_cYs[first50_ind]
                        cont_edts = tmp_edts[first50_ind]
                        cont_degs = tmp_degs[first50_ind]
                        cont_mons = tmp_mons[first50_ind]
                        print 'Sampled '+str(len(cont_cXs))+' continental cloudbands'
                        if len(cont_cXs)<50:
                            print 'LESS THAN 50 CLOUDBANDS SAMPLED'
                            exit()


                    if show_mada_sample:
                        tmp_edts = []
                        tmp_cXs = []
                        tmp_cYs = []
                        tmp_degs = []
                        tmp_mons = []

                        for b in range(len(edts)):
                            date = edts[b]
                            mon = mons[b]
                            cX = cXs[b]
                            cY = cYs[b]
                            deg = degs[b]

                            # Check on the longitude of centroid
                            if cX > w_cen[1] and cX < e_cen[1]:

                                # Check on the angle
                                if deg > t_ang[1] and deg < b_ang[1]:

                                    # Check on the month
                                    if mon >= f_seas[1] or mon <= l_seas[1]:
                                        tmp_edts.append(date)
                                        tmp_cXs.append(cX)
                                        tmp_cYs.append(cY)
                                        tmp_degs.append(deg)
                                        tmp_mons.append(mon)

                        tmp_edts = np.asarray(tmp_edts)
                        tmp_cXs = np.asarray(tmp_cXs)
                        tmp_cYs = np.asarray(tmp_cYs)
                        tmp_degs = np.asarray(tmp_degs)
                        tmp_mons = np.asarray(tmp_mons)
                        print 'Shortlist of '+str(len(tmp_edts))+' madagascan cloudbands'

                        # Get the 50 closest to best_lon
                        dists = best_lat[1] - tmp_cYs
                        abs_dists = np.absolute(dists)
                        inds = np.argsort(abs_dists)
                        dists_sort = abs_dists[inds]
                        first50_ind = inds[0:ndays[1]]

                        mada_cXs = tmp_cXs[first50_ind]
                        mada_cYs = tmp_cYs[first50_ind]
                        mada_edts = tmp_edts[first50_ind]
                        mada_degs = tmp_degs[first50_ind]
                        mada_mons = tmp_mons[first50_ind]
                        print 'Sampled '+str(len(mada_cXs))+' madagascan cloudbands'
                        if len(mada_cXs)<50:
                            print 'LESS THAN 50 CLOUDBANDS SAMPLED'
                            exit()

                if sample=='bang':
                    if show_cont_sample:
                        tmp_edts=[]
                        tmp_cXs=[]
                        tmp_cYs=[]
                        tmp_degs=[]
                        tmp_mons=[]

                        for b in range(len(edts)):
                            date=edts[b]
                            mon=mons[b]
                            cX=cXs[b]
                            cY=cYs[b]
                            deg=degs[b]

                            # Check on the latitude of centroid
                            if cY > s_cen[0] and cY < n_cen[0]:

                                # Check on the longitude of centroid
                                if cX > w_cen[0] and cX < e_cen[0]:

                                    # Check on the month
                                    if mon >= f_seas[0] or mon <= l_seas[0]:

                                        tmp_edts.append(date)
                                        tmp_cXs.append(cX)
                                        tmp_cYs.append(cY)
                                        tmp_degs.append(deg)
                                        tmp_mons.append(mon)

                        tmp_edts=np.asarray(tmp_edts)
                        tmp_cXs=np.asarray(tmp_cXs)
                        tmp_cYs=np.asarray(tmp_cYs)
                        tmp_degs=np.asarray(tmp_degs)
                        tmp_mons=np.asarray(tmp_mons)
                        print 'Shortlist of '+str(len(tmp_edts))+' continental cloudbands'

                        # Get the 50 closest to best_lon
                        dists = best_deg[0] - tmp_degs
                        abs_dists = np.absolute(dists)
                        inds = np.argsort(abs_dists)
                        dists_sort = abs_dists[inds]
                        first50_ind = inds[0:ndays[0]]

                        cont_cXs=tmp_cXs[first50_ind]
                        cont_cYs=tmp_cYs[first50_ind]
                        cont_edts=tmp_edts[first50_ind]
                        cont_degs=tmp_degs[first50_ind]
                        cont_mons=tmp_mons[first50_ind]
                        print 'Sampled '+str(len(cont_cXs))+' continental cloudbands'
                        if len(cont_cXs)<50:
                            print 'LESS THAN 50 CLOUDBANDS SAMPLED'
                            exit()


                    if show_mada_sample:
                        tmp_edts = []
                        tmp_cXs = []
                        tmp_cYs = []
                        tmp_degs = []
                        tmp_mons = []

                        for b in range(len(edts)):
                            date = edts[b]
                            mon = mons[b]
                            cX = cXs[b]
                            cY = cYs[b]
                            deg = degs[b]

                            # Check on the latitude of centroid
                            if cY > s_cen[1] and cY < n_cen[1]:

                                # Check on the longitude of centroid
                                if cX > w_cen[1] and cX < e_cen[1]:

                                    # Check on the month
                                    if mon >= f_seas[1] or mon <= l_seas[1]:
                                        tmp_edts.append(date)
                                        tmp_cXs.append(cX)
                                        tmp_cYs.append(cY)
                                        tmp_degs.append(deg)
                                        tmp_mons.append(mon)

                        tmp_edts = np.asarray(tmp_edts)
                        tmp_cXs = np.asarray(tmp_cXs)
                        tmp_cYs = np.asarray(tmp_cYs)
                        tmp_degs = np.asarray(tmp_degs)
                        tmp_mons = np.asarray(tmp_mons)
                        print 'Shortlist of '+str(len(tmp_edts))+' madagascan cloudbands'

                        # Get the 50 closest to best_lon
                        dists = best_deg[1] - tmp_degs
                        abs_dists = np.absolute(dists)
                        inds = np.argsort(abs_dists)
                        dists_sort = abs_dists[inds]
                        first50_ind = inds[0:ndays[1]]

                        mada_cXs = tmp_cXs[first50_ind]
                        mada_cYs = tmp_cYs[first50_ind]
                        mada_edts = tmp_edts[first50_ind]
                        mada_degs = tmp_degs[first50_ind]
                        mada_mons = tmp_mons[first50_ind]
                        print 'Sampled '+str(len(mada_cXs))+' madagascan cloudbands'
                        if len(mada_cXs)<50:
                            print 'LESS THAN 50 CLOUDBANDS SAMPLED'
                            exit()

            print "Plotting for model " + name

            if cenlonplot:
                plt.figure(num='cenlon')
                plt.subplot(yplots, xplots, cnt)
                y, binEdges = np.histogram(cXs, bins=25, density=True)
                bincentres = 0.5 * (binEdges[1:] + binEdges[:-1])
                plt.plot(bincentres, y)
                plt.xlim(7.5, 100.0)
                plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

            if cenlatplot:
                plt.figure(num='cenlat')
                plt.subplot(yplots, xplots, cnt)
                y, binEdges = np.histogram(cYs, bins=25, density=True)
                bincentres = 0.5 * (binEdges[1:] + binEdges[:-1])
                plt.plot(bincentres, y)
                plt.xlim(-15.0, -40.0)

            if angleplot:
                plt.figure(num='angle')
                plt.subplot(yplots, xplots, cnt)
                y, binEdges = np.histogram(degs, bins=25, density=True)
                bincentres = 0.5 * (binEdges[1:] + binEdges[:-1])
                plt.plot(bincentres, y)
                plt.xlim(-90.0, -5.0)

            if monplot:
                plt.figure(num='mon')
                plt.subplot(yplots, xplots, cnt)
                season = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]
                mon_count=np.zeros(len(season))
                for imn in xrange(len(mon_count)):
                    mn=season[imn]
                    ix = np.where(mons==mn)[0]
                    mon_count[imn]=len(ix)
                seaspos=np.arange(1,13,1)
                plt.plot(seaspos,mon_count)
                plt.xticks(np.arange(1, 13), monthstr, fontsize=13.0)  # month labels
                plt.xlim(1, 12)

            if scatter_lon_angle:
                plt.figure(num='lon_ang')
                ax = plt.subplot(yplots, xplots, cnt)
                plt.scatter(cXs,degs,c='k',marker="o",s=0.1,edgecolors='face')
                if show_sample:
                    if show_cont_sample:
                        plt.scatter(cont_cXs,cont_degs,c='fuchsia',marker="o",s=0.5,edgecolors='face')
                    if show_mada_sample:
                        plt.scatter(mada_cXs,mada_degs,c='blue',marker="o",s=0.5,edgecolors='face')
                plt.xlim(7.5, 100.0)
                xlabs=[20,40,60,80]
                plt.xticks(xlabs,fontsize=6,fontweight='demibold')
                plt.ylim(-90.0, -5.0)
                ylabs=[-90,-60,-30]
                if group:
                    for axis in ['top', 'bottom', 'left', 'right']:
                        ax.spines[axis].set_linewidth(3)
                        ax.spines[axis].set_color(grcl)
                plt.yticks(ylabs,fontsize=6, fontweight='demibold')
                plt.title(name+tag, fontsize=8, fontweight='demibold')


            if scatter_lat_angle:
                plt.figure(num='lat_ang')
                ax = plt.subplot(yplots, xplots, cnt)
                plt.scatter(degs,cYs,c='k',marker="o",s=0.1,edgecolors='face')
                if show_sample:
                    if show_cont_sample:
                        plt.scatter(cont_degs,cont_cYs,c='fuchsia',marker="o",s=0.5,edgecolors='face')
                    if show_mada_sample:
                        plt.scatter(mada_degs,mada_cYs,c='blue',marker="o",s=0.5,edgecolors='face')
                plt.xlim(-90.0, -5.0)
                xlabs=[-90,-60,-30]
                plt.xticks(xlabs,fontsize=6)
                plt.ylim(-40.0, -15.0)
                ylabs=[-40,-30,-20]
                if group:
                    for axis in ['top', 'bottom', 'left', 'right']:
                        ax.spines[axis].set_linewidth(3)
                        ax.spines[axis].set_color(grcl)
                plt.yticks(ylabs,fontsize=6)
                plt.title(name+tag, fontsize=8)

            if scatter_lon_lat:
                plt.figure(num='lon_lat')
                ax=plt.subplot(yplots, xplots, cnt)
                plt.scatter(cXs,cYs,c='k',marker="o",s=0.1,edgecolors='face')
                if show_sample:
                    if show_cont_sample:
                        plt.scatter(cont_cXs,cont_cYs,c='fuchsia',marker="o",s=0.5,edgecolors='face')
                    if show_mada_sample:
                        plt.scatter(mada_cXs,mada_cYs,c='blue',marker="o",s=0.5,edgecolors='face')
                plt.xlim(7.5, 100.0)
                xlabs=[20,40,60,80]
                plt.xticks(xlabs,fontsize=6)
                plt.ylim(-40.0, -15.0)
                ylabs=[-40,-30,-20]
                if group:
                    for axis in ['top', 'bottom', 'left', 'right']:
                        ax.spines[axis].set_linewidth(3)
                        ax.spines[axis].set_color(grcl)
                plt.yticks(ylabs,fontsize=6)
                plt.title(name+tag, fontsize=8)

            mon4scatter=np.zeros(len(mons))
            for mn in range(len(mon4scatter)):
                if mons[mn] <=7:
                    mon4scatter[mn]=mons[mn]+5
                elif mons[mn] >=8:
                    mon4scatter[mn]=mons[mn]-7

            if scatter_lon_mon:
                plt.figure(num='lon_mon')
                plt.subplot(yplots, xplots, cnt)
                plt.scatter(cXs,mon4scatter,c='k',marker="o",s=5,edgecolors='face')
                plt.xlim(7.5, 100.0)
                plt.yticks(np.arange(1, 13), monthstr, fontsize=13.0)  # month labels

            if scatter_lat_mon:
                plt.figure(num='lat_mon')
                plt.subplot(yplots, xplots, cnt)
                plt.scatter(mon4scatter,cYs,c='k',marker="o",s=5,edgecolors='face')
                plt.xticks(np.arange(1, 13), monthstr, fontsize=13.0)  # month labels
                plt.ylim(-40.0, -15.0)

            if scatter_ang_mon:
                plt.figure(num='ang_mon')
                plt.subplot(yplots, xplots, cnt)
                plt.scatter(degs,mon4scatter,c='k',marker="o",s=5,edgecolors='face')
                plt.xlim(-90.0, -5.0)
                plt.yticks(np.arange(1, 13), monthstr, fontsize=13.0)  # month labels


            cnt += 1

            print 'Finished running on ' + name
            print 'This is model '+mcnt+' of '+nmstr+' in list'

    ### Name supplement for options
    namesup="fromevent_"+from_event+"_"
    if one_seas:
        namesup=namesup+seas_pick
    if show_sample:
        namesup=namesup+"_"+sample
    if group:
        namesup=namesup+"_grouped"

    ### Save figure
    if cenlonplot:
        plt.figure(num='cenlon')
        cenlonfig=figdir+'/hist_cen_lon.'+thnames[t]+'.'+namesup+'.png'
        plt.savefig(cenlonfig)

    if cenlatplot:
        plt.figure(num='cenlat')
        cenlatfig = figdir + '/hist_cen_lat.'+thnames[t]+'.'+namesup+'.png'
        plt.savefig(cenlatfig)

    if angleplot:
        plt.figure(num='angle')
        anglefig = figdir + '/hist_angle.'+thnames[t]+'.'+namesup+'.png'
        plt.savefig(anglefig)

    if monplot:
        plt.figure(num='mon')
        monfig = figdir + '/hist_mon.'+thnames[t]+'.'+namesup+'.png'
        plt.savefig(monfig)

    if scatter_lon_angle:
        plt.figure(num='lon_ang')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.02, wspace=0.2, hspace=0.3)
        lonangfig = figdir + '/scatter_lon_angle.'+thnames[t]+'.'+namesup+'.png'
        plt.savefig(lonangfig)

    if scatter_lat_angle:
        plt.figure(num='lat_ang')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.02, wspace=0.2, hspace=0.3)
        latangfig = figdir + '/scatter_lat_angle.'+thnames[t]+'.'+namesup+'.png'
        plt.savefig(latangfig)

    if scatter_lon_lat:
        plt.figure(num='lon_lat')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.02, wspace=0.2, hspace=0.3)
        lonlatfig = figdir + '/scatter_lon_lat.'+thnames[t]+'.'+namesup+'.png'
        plt.savefig(lonlatfig)

    if scatter_lon_mon:
        plt.figure(num='lon_mon')
        lonmonfig = figdir + '/scatter_lon_mon.'+thnames[t]+'.'+namesup+'.png'
        plt.savefig(lonmonfig)

    if scatter_lat_mon:
        plt.figure(num='lat_mon')
        latmonfig = figdir + '/scatter_lat_mon.'+thnames[t]+'.'+namesup+'.png'
        plt.savefig(latmonfig)

    if scatter_ang_mon:
        plt.figure(num='ang_mon')
        angmonfig = figdir + '/scatter_ang_mon.'+thnames[t]+'.'+namesup+'.png'
        plt.savefig(angmonfig)

    plt.close('all')

    print 'Finished running on ' + thre_str
    print 'This is thresh ' + str(t) + ' of ' + str(nthresh) + ' in list'
