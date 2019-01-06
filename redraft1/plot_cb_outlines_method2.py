# To show variation within model composites
# by plotting the outline of the CB
# this is also a check that I am selecting them correctly in ANOVA analysis
# and potentially a way to figure out at which latitudes or longitudes there might be more or less similarity

# METHOD 2 - previously I had done this by selecting the date and then the CB
# looked quite wild and I think could be because some dates have more than one CB
# so now let's extract them directly as I did originally


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



### Running options
test_scr=False
cb_test=False # start by just plotting one CB

plotshow='greyall' # 'greyall' or 'col5'

xplots = 4
yplots = 7

runs=['opt1']
wcb=['cont','mada'] # which cloud band composite? Options: cont, mada, dbl
thname='actual'
alphord=True
domain='swio'
seas='NDJFM'
refkey='0'              # for MetBot


## Info for options
if domain=='swio':
    sub='SA'
    figdim=[9,11]

import dsets_paper_28_4plot as dset_mp

### Get directories
bkdir=cwd+"/../../../CTdata/"
thisdir=bkdir+"/hpaperplot/"
botdir=bkdir+"metbot_multi_dset/"

plotdir=thisdir+"cb_outlines/"
my.mkdir_p(plotdir)

if seas == 'NDJFM':
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

    # Loop sample type
    for o in range(len(wcb)):
        type = wcb[o]
        if type=='cont':
            sind=0
        elif type=='mada':
            sind=1

        # Set up plot
        print "Setting up plot..."
        g, ax = plt.subplots(figsize=figdim)

        cnt = 1

        ### Dsets
        if test_scr:
            dsets='spec'
            ndset=1
            dsetnames=['noaa']
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

        print "Looping datasets"
        for d in range(ndset):
            dset=dsetnames[d]
            dcnt=str(d+1)
            print 'Running on '+dset
            print 'This is dset '+dcnt+' of '+ndstr+' in list'

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

                # Get info
                moddct = dsetdict.dset_deets[dset][name]
                globv='olr'
                vnamedict = globv + 'name'
                mastdct = mast_dict.mast_dset_deets[dset]
                varstr = mastdct[vnamedict]
                dimdict = dim_exdict.dim_deets[globv][dset]
                latname = dimdict[1]
                lonname = dimdict[2]
                ys = moddct['yrfname']

                # Open file
                botpath=bkdir + 'metbot_multi_dset/' + dset + '/' + name + '/'
                smpfile=botpath+name+'.'+name+'.'+globv+'.sampled_days.'\
                     +sample+'.'+from_event+'.'+type+'.'+thname+'.nc'

                print 'Opening '+smpfile

                ncout = mync.open_multi(smpfile, globv, name, \
                                         dataset=dset,subs=sub)
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

                print 'opening metbot files...'
                botpath=bkdir + 'metbot_multi_dset/' + dset + '/' + name + '/'
                outsuf = botpath + name + '_'
                syfile = outsuf + thre_str + '_' + dset + '-OLR.synop'
                mbsfile = outsuf + thre_str + '_' + dset + "-olr-0-0.mbs"
                refmbs, refmbt, refch = blb.mbopen(mbsfile)
                refmbt[:, 3] = 0
                s = sy.SynopticEvents((), [syfile], COL=False)
                ks = s.events.keys();ks.sort()  # all
                refkey = s.mbskeys[0]

                # Select the first day
                if from_event=='first':
                    ev_dts=[]
                    ev_keys=[]
                    ev_cXs=[]
                    ev_chs=[]

                    for k in ks:
                        e = s.events[k]
                        dts = s.blobs[refkey]['mbt'][e.ixflags]
                        for dt in range(len(dts)):
                            x,y = e.trkcX[dt], e.trkcY[dt]
                            ev_dts.append(dts[dt])
                            ev_keys.append(k)
                            ev_cXs.append(x)
                            ev_chs.append(e.blobs[refkey]['ch'][e.trk[dt]])


                    ev_dts=np.asarray(ev_dts)
                    ev_dts[:,3]=0
                    ev_keys=np.asarray(ev_keys)
                    ev_cXs=np.asarray(ev_cXs)
                    ev_chs=np.asarray(ev_chs)

                ### Get array of centroids and angles and chs
                edts = []
                cXs = []
                cYs = []
                degs = []
                mons = []
                chs = []

                for b in range(len(refmbt)):
                    date = refmbt[b]
                    mon = int(date[1])
                    cX = refmbs[b, 3]
                    cY = refmbs[b, 4]
                    deg = refmbs[b, 2]

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


                ### Limit to certain mons
                if seas == 'NDJFM':  # because the > and < statement below only works for summer
                    pick1 = np.where(mons >= f_mon)[0]
                    pick2 = np.where(mons <= l_mon)[0]
                    pick = np.concatenate([pick1, pick2])
                    edts = edts[pick]
                    cXs = cXs[pick]
                    cYs = cYs[pick]
                    degs = degs[pick]
                    mons = mons[pick]
                    chs = chs[pick]

                ### Find sample
                if sample == 'blon' or sample == 'blon2':
                    tmp_edts = []
                    tmp_cXs = []
                    tmp_cYs = []
                    tmp_degs = []
                    tmp_mons = []
                    tmp_chs = []

                    for b in range(len(edts)):
                        date = edts[b]
                        mon = mons[b]
                        cX = cXs[b]
                        cY = cYs[b]
                        deg = degs[b]
                        ch = chs[b]

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

                    tmp_edts = np.asarray(tmp_edts)
                    tmp_cXs = np.asarray(tmp_cXs)
                    tmp_cYs = np.asarray(tmp_cYs)
                    tmp_degs = np.asarray(tmp_degs)
                    tmp_mons = np.asarray(tmp_mons)
                    tmp_chs = np.asarray(tmp_chs)
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
                    print 'Sampled ' + str(len(smp_cXs)) + ' cloudbands'
                    if len(smp_cXs) < 50:
                        print 'LESS THAN 50 CLOUDBANDS SAMPLED'
                        tag = ' ONLY_' + str(len(smp_cXs))
                    else:
                        tag = ''


                # Make a map
                plt.subplot(yplots, xplots, cnt)
                print 'Generating map'
                m, mfig = pt.AfrBasemap(lat, lon, drawstuff=True, prj='cyl', fno=1, rsltn='l')

                nch=len(smp_chs)
                if plotshow=='col5':
                    nch=5
                    cols = ['r', 'b', 'c', 'm', 'g', 'r', 'b', 'c', 'm', 'g']
                if cb_test:
                    nch=1

                print 'Looping sample days and plotting CB outlines'
                for jl in range(nch):
                    cb=smp_chs[jl]
                    if plotshow=='greyall':
                        cl='darkgray'
                    else:
                        cl=cols[jl]
                    cnx, cny = cb[:, 0], cb[:, 1]
                    m.plot(cnx,cny,cl,lw=1.)

                # Redraw map
                m.drawcountries()
                m.drawcoastlines()
                # m.drawparallels(np.arange(-90.,90.0,20.0),linewidth='0',labels=[1,0,0,0])
                # m.drawmeridians(np.arange(0.,360.0,30.0),linewidth='0',labels=[0,0,0,1])

                plt.title(name, fontsize=8, fontweight='demibold')

                cnt +=1

        print "Finalising plot..."
        plt.subplots_adjust(left=0.05,right=0.9,top=0.95,bottom=0.02,wspace=0.1,hspace=0.2)

        # Saving output plot
        if test_scr:
            mods='testmodels'
        else:
            mods='allmod'

        end=''
        if cb_test:
            end='_test.'

        compname= plotdir + 'cb_outlines_plot.method2.'+plotshow+'.sample_'+type+'.'+globv+'.models_'+mods+'.'+end+'png'
        plt.savefig(compname, dpi=150)
        plt.close()