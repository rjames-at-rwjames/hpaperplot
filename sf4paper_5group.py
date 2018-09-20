# To plot all CMIP5 models in multi-panel plot
# gridpoint frequency maps
#
# OLR threshold is detected automatically using "find_saddle"
# Option to run on other OLR thresholds a test - currently + and - 5Wm2

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

cwd=os.getcwd()
sys.path.append(cwd+'/..')
sys.path.append(cwd+'/../MetBot')
sys.path.append(cwd+'/../RTools')
import MetBot.SynopticAnatomy as sy
import MetBot.EventStats as stats
import MetBot.MetBlobs as blb
import MetBot.mytools as my
import MetBot.mynetcdf as mync
import MetBot.dset_dict as dsetdict
import dsets_mplot_5group_4plot as dset_mp


### Running options
sub="SA_TR"
#seasons=['DJ','DJF','NDJFM']
seasons=['NDJFM']
runs=['opt1'] # sample for centroids that go on plot
rate='cbs' # if rate='year' it will plot cbs per year
            # if cbs it will plot for that models total number of CBs

res='noaa'              # Option to plot at 'noaa' res or 'native' res

threshtest=True # Option to run on thresholds + and - 5Wm2 as a test
col_plot=True
bwcen_plot=False
group=True
future=False
year1=2065
year2=2099

xplots = 4
yplots = 7

### Get directories
bkdir=cwd+"/../../CTdata/"
thisdir=bkdir+"hpaperplot/"
botdir=bkdir+"metbot_multi_dset/"

figdir=thisdir+"allCMIPplot_spatiofreq/"
my.mkdir_p(figdir)


if group:
    grcls=['fuchsia','gold','darkblue','r','blueviolet','springgreen']

# Loop sampling options
for r in range(len(runs)):

    if runs[r]=='opt1':
        sample='blon'
        from_event='first'
    elif runs[r]=='opt2':
        sample='blon2'
        from_event='all'

    for s in range(len(seasons)):

        seas=seasons[s]

        if seas=='NDJFM':
            mon1=11
            mon2=3
            nmon=5
            if rate=='year':
                nos4cbar=(4,52,4)        # choose the intervals for spatiofreq cbar

        if seas=='DJF':
            mon1=12
            mon2=2
            nmon=3
            if rate=='year':
                nos4cbar = (3, 39, 3)

        if seas=='DJ':
            mon1=12
            mon2=1
            nmon=2
            if rate=='year':
                nos4cbar = (2, 26, 2)

        if rate=='cbs':
            nos4cbar= (20,50,3)

        ### Loop threshs
        if threshtest:
            thnames = ['lower', 'actual', 'upper']
        else:
            thnames = ['actual']

        nthresh = len(thnames)
        for t in range(nthresh):

            # Set up plot
            print "Setting up plot..."

            if col_plot: plt.figure(num='col',figsize=[10,7])
            if bwcen_plot: plt.figure(num='bw',figsize=[10,7])

            cnt = 1

            # Get the map
            if cnt==1:
                noaadct = dsetdict.dset_deets['noaa']['noaa']
                yr_noaa = noaadct['yrfname']
                f_noaa = botdir + \
                         "noaa/noaa.olr.day.mean." + yr_noaa + ".nc"
                olrdump, timedump, noaalat, noaalon, dtimedump = mync.openolr(f_noaa, 'olr', subs=sub)


                m = blb.SAfrBasemap2(noaalat, noaalon, drawstuff=True, prj='cyl',
                                       rsltn='l')

            ### Dsets
            dsets = 'all'
            ndset = len(dset_mp.dset_deets)
            # dsetnames=list(dset_mp.dset_deets)
            dsetnames = ['noaa', 'cmip5']
            # dsets='spec'
            # ndset=1
            # dsetnames=['noaa']
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

                if dset=='cmip5':
                    if group:
                        mnames=np.zeros(nmod,dtype=object)

                        for mo in range(nmod):
                            name=mnames_tmp[mo]
                            moddct = dset_mp.dset_deets[dset][name]
                            thisord=int(moddct['ord'])-2 # minus 2 because cdr already used
                            mnames[thisord]=name

                    else:
                        mnames=mnames_tmp
                else:
                    mnames=mnames_tmp


                for mo in range(nmod):
                    name = mnames[mo]
                    mcnt = str(mo + 1)
                    print 'Running on ' + name
                    print 'This is model ' + mcnt + ' of ' + nmstr + ' in list'

                    if group:
                        groupdct=dset_mp.dset_deets[dset][name]
                        thisgroup = int(groupdct['group'])
                        grcl = grcls[thisgroup - 1]

                    if res=='noaa':
                        lat=noaalat
                        lon=noaalon

                    elif res=='native':

                        # Get details
                        moddct=dsetdict.dset_deets[dset][name]
                        vname=moddct['olrname']
                        ys=moddct['yrfname']

                        ### Location for olr input & outputs
                        indir=botdir+"/"+dset+"/"
                        infile=indir+name+".olr.day.mean."+ys+".nc"
                        print infile

                        ### Open olr nc file
                        v = dset + "-olr-0-0"
                        daset, globv, lev, drv = v.split('-')
                        ncout = mync.open_multi(infile,globv,name,\
                                                                    dataset=dset,subs=sub)
                        ndim = len(ncout)
                        if ndim == 5:
                            olr, time, lat, lon, dtime = ncout
                        elif ndim == 6:
                            olr, time, lat, lon, lev, dtime = ncout
                            olr = np.squeeze(olr)
                        else:
                            print 'Check number of levels in ncfile'

                    ### Get threshold
                    threshtxt = botdir + 'thresholds.fmin.all_dset.txt'
                    print 'using metbot output'
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
                        thisthresh = thresh - 5
                    if thnames[t] == 'upper':
                        thisthresh = thresh + 5

                    thre_str = str(int(thisthresh))

                    sydir = botdir + dset + '/' + name + '/'
                    sysuf = sydir + name + '_'
                    if future:
                        sysuf = sysuf + 'fut_'
                    syfile=sysuf+thre_str+'_'+dset+'-OLR.synop'

                    if os.path.exists(syfile):

                        ### Open ttt data
                        s = sy.SynopticEvents((),[syfile],COL=False)


                        ### Select events
                        ks = s.events.keys();ks.sort() # all
                        refkey='0'
                        key= dset + '-olr-0-' + refkey

                        edts=[]
                        thesekeys=[]
                        for k in ks:
                            e = s.events[k]
                            dts = s.blobs[key]['mbt'][e.ixflags]
                            if len(dts) > 1:
                                dt = dts[len(dts) / 2]
                            else:
                                dt = dts[0]
                            if (int(dt[1]) >= mon1) or (int(dt[1]) <= mon2):
                                if future:
                                    if (int(dt[0]) >= year1) or (int(dt[0]) <= year2):
                                        thesekeys.append(k)
                                        edts.append(dt)
                                else:
                                    thesekeys.append(k)
                                    edts.append(dt)
                        edts=np.asarray(edts)
                        yrs=np.unique(edts[:,0])

                        ### PLOT GRIDPOINT COUNT FOR WHOLE SEASON
                        print 'Plotting spatiofrequency for model '+name

                        # colour plot
                        if col_plot:
                            print 'Plotting spatiofreq plot in colour'
                            plt.figure(num='col')
                            plt.subplot(yplots,xplots,cnt)
                            allmask=stats.spatiofreq4(m,s,name,lat,lon,yrs,thesekeys,per=rate,meanmask=False,clim=nos4cbar,\
                                                  month=False,savefig=False,flagonly=True,\
                                                      col='col',cens='None',frm_event='all')
                            m.drawcountries(color='k')
                            m.drawcoastlines(color='k')
                            if group:
                                m.drawmapboundary(color=grcl,linewidth=3)

                        if bwcen_plot:
                            print 'Getting sample to plot centroids on b/w spatiofreq'

                            if sample == 'blon':
                                best_lon = [33, 58]
                                ndays = [50, 50]

                                n_cen = [-22, -22]
                                s_cen = [-32, -32]

                                t_ang = [-60, -50]
                                b_ang = [-25, -15]

                                f_seas = [11, 11]
                                l_seas = [3, 3]

                            if sample == 'blon2':
                                best_lon = [33, 60]
                                ndays = [50, 50]

                                n_cen = [-26, -24]
                                s_cen = [-30, -28]

                                t_ang = [-50, -40]
                                b_ang = [-35, -25]

                                f_seas = [11, 11]
                                l_seas = [3, 3]

                            # Open mbs
                            print 'Opening mbs file'
                            mbsfile = sysuf + thre_str + '_' + dset + "-olr-0-0.mbs"
                            refmbs, refmbt, refch = blb.mbopen(mbsfile)
                            refmbt[:, 3] = 0
                            refkey = s.mbskeys[0]

                            # Count number of events
                            if from_event == 'first':
                                print 'If sample is from first day of event getting events'

                                ev_dts = []
                                ev_keys = []
                                ev_cXs = []

                                for k in thesekeys:
                                    e = s.events[k]
                                    dts = s.blobs[refkey]['mbt'][e.ixflags]
                                    for dt in range(len(dts)):
                                        x, y = e.trkcX[dt], e.trkcY[dt]
                                        ev_dts.append(dts[dt])
                                        ev_keys.append(k)
                                        ev_cXs.append(x)

                                ev_dts = np.asarray(ev_dts)
                                ev_dts[:, 3] = 0
                                ev_keys = np.asarray(ev_keys)
                                ev_cXs = np.asarray(ev_cXs)

                            ### Get array of centroids and angles
                            print 'Getting an array of all CB blob info'
                            edts = []
                            cXs = []
                            cYs = []
                            degs = []
                            mons = []

                            for b in range(len(refmbt)):
                                date = refmbt[b]
                                yr = int(date[0])
                                mon = int(date[1])
                                cX = refmbs[b, 3]
                                cY = refmbs[b, 4]
                                deg = refmbs[b, 2]

                                if from_event == 'all':
                                    if future:
                                        if (yr >= year1) or (yr <= year2):
                                            edts.append(date)
                                            cXs.append(cX)
                                            cYs.append(cY)
                                            degs.append(deg)
                                            mons.append(mon)
                                            years.append(yr)
                                    else:
                                        edts.append(date)
                                        cXs.append(cX)
                                        cYs.append(cY)
                                        degs.append(deg)
                                        mons.append(mon)
                                        years.append(yr)
                                elif from_event == 'first':
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
                                                    edts.append(date)
                                                    cXs.append(cX)
                                                    cYs.append(cY)
                                                    degs.append(deg)
                                                    mons.append(mon)

                                    elif len(ix) > 1:
                                        todays_cXs = ev_cXs[ix]
                                        index2 = np.where(todays_cXs == cX)[0]
                                        if len(index2) != 1:
                                            print 'Error - centroid not matching'
                                        index3 = ix[index2]
                                        key = ev_keys[index3]
                                        e = s.events[key[0]]
                                        dts = s.blobs[refkey]['mbt'][e.ixflags]
                                        if dts[0, 0] == date[0]:
                                            if dts[0, 1] == date[1]:
                                                if dts[0, 2] == date[2]:
                                                    edts.append(date)
                                                    cXs.append(cX)
                                                    cYs.append(cY)
                                                    degs.append(deg)
                                                    mons.append(mon)

                            edts = np.asarray(edts)
                            edts[:, 3] = 0
                            cXs = np.asarray(cXs)
                            cYs = np.asarray(cYs)
                            degs = np.asarray(degs)
                            mons = np.asarray(mons)

                            ### Limit to certain mons
                            print 'Limiting to certain months'
                            pick1 = np.where(mons >= f_seas[0])[0]
                            pick2 = np.where(mons <= l_seas[0])[0]
                            pick = np.concatenate([pick1, pick2])
                            edts = edts[pick]
                            cXs = cXs[pick]
                            cYs = cYs[pick]
                            degs = degs[pick]
                            mons = mons[pick]

                            # Loop cont and mada to get cX and cY
                            wcb=['cont','mada']
                            cen_lons=np.zeros((ndays[0],2),dtype=np.float32)
                            cen_lats=np.zeros((ndays[0],2),dtype=np.float32)
                            for o in range(len(wcb)):
                                type = wcb[o]
                                print "Extracting centroids for sample " + type
                                if type=='cont':
                                    jj=0
                                elif type=='mada':
                                    jj=1

                                tmp_edts = []
                                tmp_cXs = []
                                tmp_cYs = []

                                for b in range(len(edts)):
                                    date = edts[b]
                                    mon = mons[b]
                                    cX = cXs[b]
                                    cY = cYs[b]
                                    deg = degs[b]

                                    # Check on the latitude of centroid
                                    if cY > s_cen[jj] and cY < n_cen[jj]:

                                        # Check on the angle
                                        if deg > t_ang[jj] and deg < b_ang[jj]:
                                            tmp_edts.append(date)
                                            tmp_cXs.append(cX)
                                            tmp_cYs.append(cY)

                                tmp_edts = np.asarray(tmp_edts)
                                tmp_edts[:, 3] = 0
                                tmp_cXs = np.asarray(tmp_cXs)
                                tmp_cYs = np.asarray(tmp_cYs)
                                dists = best_lon[jj] - tmp_cXs

                                abs_dists = np.absolute(dists)
                                inds = np.argsort(abs_dists)
                                dists_sort = abs_dists[inds]
                                first50_ind = inds[0:ndays[jj]]

                                edts_50 = tmp_edts[first50_ind]
                                cXs_50 = tmp_cXs[first50_ind]
                                cYs_50 = tmp_cYs[first50_ind]

                                cen_lons[:,o]=cXs_50
                                cen_lats[:,o]=cYs_50


                            cont_cens=np.zeros((ndays[0],2),dtype=np.float32)
                            mada_cens=np.zeros((ndays[0],2),dtype=np.float32)

                            cont_cens[:,0]=cen_lons[:,0]
                            cont_cens[:,1]=cen_lats[:,1]

                            mada_cens[:,0]=cen_lons[:,1]
                            mada_cens[:,1]=cen_lats[:,1]

                            print 'Ready to plot b/w centroid figure'
                            plt.figure(num='bw')
                            plt.subplot(yplots,xplots,cnt)
                            # allmask=stats.spatiofreq4(m,s,name,lat,lon,yrs,thesekeys,per=rate,meanmask=False, clim=nos4cbar,\
                            #                       month=False,savefig=False,flagonly=True,\
                            #                            col='bw',cens='all',frm_event='all')
                            allmask=stats.spatiofreq4(m,s,name,lat,lon,yrs,thesekeys,per=rate,meanmask=False, clim=nos4cbar,\
                                                  month=False,savefig=False,flagonly=True,\
                                                      col='bw',cens=[cont_cens, mada_cens],frm_event='all')
                            m.drawcountries(color='k')
                            m.drawcoastlines(color='k')
                            if group:
                                m.drawmapboundary(color=grcl,linewidth=3)


                    cnt+=1

            # Final stuff
            if group:
                figsuf='grouped'
            else:
                figsuf=''

            if future:
                figsuf=figsuf+'_future'


            if col_plot:
                print 'Finalising colour plot'
                plt.figure(num='col')
                plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.02, wspace=0.1, hspace=0.2)

                # if group:
                    # Draw boxes


                figname = figdir + 'multi_spatiofreq.'+seas+'.'+res+'.' + sub + '.per_'+rate+'.colour.'+figsuf+'.'+thnames[t]+'.png'
                print 'saving figure as ' + figname
                plt.savefig(figname, dpi=150)
                plt.close()

            if bwcen_plot:
                print 'Finalising bw centroid plot'
                plt.figure(num='bw')
                plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.02, wspace=0.1, hspace=0.2)
                figname = figdir + 'multi_spatiofreq.'+seas+'.'+res+'.'\
                          + sub + '.per_'+rate+'.'+sample+'.'+from_event+'.bwcen.'+thnames[t]+'.png'
                print 'saving figure as ' + figname
                plt.savefig(figname, dpi=150)
                plt.close()