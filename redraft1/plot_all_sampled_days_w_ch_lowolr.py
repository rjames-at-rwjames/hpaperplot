# To plot a maps for flagged CB days
# after selection based on angle and centroid
#
# plotted over a larger domain
# with centroid and angle displayed
# NOW WITH FILTER TO ONLY SAVE THOSE WITH LOW OLR VALS

import mpl_toolkits.basemap as bm
from mpl_toolkits.basemap import Basemap, cm
from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt
import sys,os
cwd=os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd+'/../../MetBot')
sys.path.append(cwd+'/../../RTools')
sys.path.append(cwd+'/../')
import PlotTools as pt
import MetBot.dset_dict as dsetdict
import dsets_paper_28_4plot as dset_mp
import MetBot.dimensions_dict as dim_exdict
import MetBot.mytools as my
import MetBot.mynetcdf as mync
import MetBot.SynopticAnatomy as sy
import MetBot.MetBlobs as blb

import time as tm
import datetime


### Running options
size='20'
globv='olr'
postrmm=False
sub='SA'
from_event='first'
sample='blon'
type='cont'
olrfilt=True # apply filter on mean OLR under CB
howolr='botolr' # type of OLR mean filter
                # 'thlim' - all CBs with mean OLR under threshold are retained
                # 'botolr' - CBs with lowest OLR

if type == 'cont':
    jj = 0

if sample == 'blon':
    best_lon = [33,58]
    ndays=[50,50]

    n_cen = [-22,-22]
    s_cen = [-32,-32]

    t_ang = [-60,-50]
    b_ang = [-25,-15]

    f_seas = [11,11]
    l_seas = [3,3]


# How many plots do you want?
if size=='20':
    nplot=int(size)
    xplots=4
    yplots=5

### Get directories
bkdir=cwd+"/../../../CTdata/"
botdir=bkdir+"metbot_multi_dset/"
thisdir=bkdir+"hpaperplot/plot_all_sampled_days"

my.mkdir_p(thisdir)


### Multi dset?
dsets='spec'     # "all" or "spec" to choose specific dset(s)
if dsets=='all':
    ndset=len(dset_mp.dset_deets)
    dsetnames=list(dset_mp.dset_deets)
elif dsets=='spec': # edit for the dset you want
    #ndset=1
    #dsetnames=['ncep']
    ndset=2
    dsetnames=['noaa','cmip5']
ndstr=str(ndset)

for d in range(ndset):
    dset=dsetnames[d]
    dcnt=str(d+1)
    print 'Running on '+dset
    print 'This is dset '+dcnt+' of '+ndstr+' in list'

    outdir=thisdir+'/'+dset+'/'
    if olrfilt:
        outdir=outdir+howolr+'/'
    my.mkdir_p(outdir)

    ### Multi model?
    mods = 'all'  # "all" or "spec" to choose specific model(s)
    if mods == 'all':
        nmod = len(dset_mp.dset_deets[dset])
        mnames = list(dset_mp.dset_deets[dset])
    if mods == 'spec':  # edit for the models you want
        nmod = 1
        mnames = ['ACCESS1-0']
        #nmod=5
        #mnames=['ACCESS1-0','bcc-csm1-1-m','CanESM2','GFDL-CM3','MIROC-ESM']
    nmstr = str(nmod)

    for m in range(nmod):
        name = mnames[m]
        mcnt = str(m + 1)
        print 'Running on ' + name
        print 'This is model ' + mcnt + ' of ' + nmstr + ' in list'

        # Get info
        moddct = dsetdict.dset_deets[dset][name]
        vnamedict = globv + 'name'
        varstr = moddct[vnamedict]
        ys = moddct['yrfname']
        dimdict = dim_exdict.dim_deets[globv][dset]
        latname = dimdict[1]
        lonname = dimdict[2]

        # Open olr file
        olrfile=botdir+dset+'/'+name+'.'+globv+\
                '.day.mean.'+ys+'.nc'
        print 'Opening '+olrfile
        ncout = mync.open_multi(olrfile, globv, name, \
                                dataset=dset, subs=sub)
        ndim = len(ncout)
        if ndim == 5:
            olrdata, time, lat, lon, dtime = ncout
        elif ndim == 6:
            olrdata, time, lat, lon, lev, dtime = ncout
            olrdata = np.squeeze(olrdata)
        else:
            print 'Check number of dims in ncfile'
        dtime[:, 3] = 0

        # Select dates with TTCBs only
        threshtxt = botdir + 'thresholds.fmin.all_dset.txt'
        print threshtxt
        with open(threshtxt) as f:
            for line in f:
                if dset + '\t' + name in line:
                    thresh = line.split()[2]
                    print 'thresh=' + str(thresh)

        thresh = int(thresh)
        thisthresh = thresh
        thre_str = str(int(thisthresh))

        ###  Open synop file
        sydir=botdir+dset+'/'+name+'/'
        sysuf=sydir+name+'_'
        mbsfile = sysuf + thre_str + '_' + dset + "-olr-0-0.mbs"
        refmbs, refmbt, refch = blb.mbopen(mbsfile)
        refmbt[:,3]=0

        if from_event == 'first':

            syfile = sysuf + thre_str + '_' + dset + '-OLR.synop'
            s = sy.SynopticEvents((), [syfile], COL=False)
            ks = s.events.keys()
            ks.sort()
            refkey = s.mbskeys[0]
            count_all = str(int(len(ks)))
            print "Total CBs flagged =" + str(count_all)

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

        ### Get array of centroids and angles
        edts = []
        cXs = []
        cYs = []
        degs = []
        mons = []
        chs = []
        olrmns = []

        for b in range(len(refmbt)):
            date = refmbt[b]
            mon = int(date[1])
            cX = refmbs[b, 3]
            cY = refmbs[b, 4]
            deg = refmbs[b, 2]
            olrmn = refmbs[b,11]

            if from_event == 'all':
                edts.append(date)
                cXs.append(cX)
                cYs.append(cY)
                degs.append(deg)
                mons.append(mon)
                olrmns.append(olrmn)
            elif from_event == 'first':
                #                    print 'Checking if the date is the first day of an event'
                ix = my.ixdtimes(ev_dts, [date[0]], \
                                 [date[1]], [date[2]], [0])
                if len(ix) == 1:
                    key = ev_keys[ix]
                    e = s.events[key[0]]
                    dts = s.blobs[refkey]['mbt'][e.ixflags]
                    if dts[0, 0] == date[0]:
                        if dts[0, 1] == date[1]:
                            if dts[0, 2] == date[2]:
                                #                                   print 'it is the first date, so we keep it'
                                edts.append(date)
                                cXs.append(cX)
                                cYs.append(cY)
                                degs.append(deg)
                                mons.append(mon)
                                chs.append(e.blobs[refkey]['ch'][e.trk[0]])
                                olrmns.append(olrmn)

                #             else:
                #                 print 'this is not the first day... ignore'
                #         else:
                #             print 'this is not the first day... ignore'
                elif len(ix) > 1:
                    # print 'there is more than one event on this day'
                    # print 'lets find the centroid that matches'
                    todays_cXs = ev_cXs[ix]
                    index2 = np.where(todays_cXs == cX)[0]
                    if len(index2) != 1:
                        print 'Error - centroid not matching'
                    index3 = ix[index2]
                    key = ev_keys[index3]
                    # print 'selecting event with matching centroid'
                    e = s.events[key[0]]
                    dts = s.blobs[refkey]['mbt'][e.ixflags]
                    # print 'but is it the first date?'
                    if dts[0, 0] == date[0]:
                        if dts[0, 1] == date[1]:
                            if dts[0, 2] == date[2]:
                                # print 'it is the first date, so we keep it'
                                edts.append(date)
                                cXs.append(cX)
                                cYs.append(cY)
                                degs.append(deg)
                                mons.append(mon)
                                chs.append(e.blobs[refkey]['ch'][e.trk[0]])
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
        olrmns = np.asarray(olrmns)

        print 'Number of events (first days)'
        print len(edts)

        # Order by olr mean
        inds = np.argsort(olrmns)
        first50_ind = inds[0:ndays[jj]]

        # Get a key for each one so we can re-order
        shlist_len=len(edts)
        mykeys=np.arange(0,shlist_len,1)

        print mykeys

        edts_50 = edts[first50_ind]
        cXs_50 = cXs[first50_ind]
        cYs_50 = cYs[first50_ind]
        degs_50 = degs[first50_ind]
        mykeys_sel=mykeys[first50_ind]
        chs_50 = chs[first50_ind]

        print mykeys_sel

        # Reorder
        keyinds=np.argsort(mykeys_sel)
        smp_edts = edts_50[keyinds]
        smp_cXs = cXs_50[keyinds]
        smp_cYs = cYs_50[keyinds]
        smp_degs = degs_50[keyinds]
        mykeys_final = mykeys_sel[keyinds]
        smp_chs = chs_50[keyinds]

        print mykeys_final
        print smp_edts

        # Find indices from var file
        indices_m1 = []
        for e in range(len(smp_edts)):
            date = smp_edts[e]

            ix = my.ixdtimes(dtime, [date[0]], [date[1]], [date[2]], [0])
            if len(ix) >= 1:
                indices_m1.append(ix)

        indices_m1 = np.squeeze(np.asarray(indices_m1))

        # Select these dates
        olrsel = olrdata[indices_m1, :, :]
        dates = dtime[indices_m1]


        # Select the dates for 50 closest

        # ### Loop flagged days and select those with certain angle and centroid
        # print 'looping flagged days to find good centroids and angles'
        # tmp_edts = []
        # if sample == 'blon' or sample == 'blon2':
        #     tmp_cXs = []
        #
        # tmp_cYs = []
        # tmp_degs = []
        # tmp_mons = []
        # tmp_chs = []
        # tmp_olrmns = []
        #
        # for b in range(len(edts)):
        #     date = edts[b]
        #     mon = mons[b]
        #     cX = cXs[b]
        #     cY = cYs[b]
        #     deg = degs[b]
        #     ch = chs[b]
        #     olrmn = olrmns[b]
        #
        #     # Check on the month
        #     if mon >= f_seas[jj] or mon <= l_seas[jj]:
        #
        #         if sample == 'blon' or sample == 'blon2':
        #
        #             # Check on the latitude of centroid
        #             if cY > s_cen[jj] and cY < n_cen[jj]:
        #
        #                 # Check on the angle
        #                 if deg > t_ang[jj] and deg < b_ang[jj]:
        #
        #                     # If olr filter apply
        #                     if olrfilt:
        #                         if howolr=='thlim':
        #                             if olrmn < thresh:
        #                                 tmp_edts.append(date)
        #                                 tmp_cXs.append(cX)
        #                                 tmp_cYs.append(cY)
        #                                 tmp_degs.append(deg)
        #                                 tmp_mons.append(mon)
        #                                 tmp_chs.append(ch)
        #                                 tmp_olrmns.append(olrmn)
        #
        #                     else:
        #                         tmp_edts.append(date)
        #                         tmp_cXs.append(cX)
        #                         tmp_cYs.append(cY)
        #                         tmp_degs.append(deg)
        #                         tmp_mons.append(mon)
        #                         tmp_chs.append(ch)
        #                         tmp_olrmns.append(olrmn)
        #
        # tmp_edts = np.asarray(tmp_edts)
        # tmp_edts[:, 3] = 0
        # if sample == 'blon' or sample == 'blon2':
        #     tmp_cXs = np.asarray(tmp_cXs)
        #     dists = best_lon[jj] - tmp_cXs
        # tmp_cYs = np.asarray(tmp_cYs)
        # tmp_degs = np.asarray(tmp_degs)
        # tmp_mons = np.asarray(tmp_mons)
        # tmp_chs = np.asarray(tmp_chs)
        # tmp_olrmns = np.asarray(tmp_olrmns)

        # # Order distance from centroid
        # abs_dists = np.absolute(dists)
        # inds = np.argsort(abs_dists)
        # dists_sort = abs_dists[inds]
        # first50_ind = inds[0:ndays[jj]]

        # # # Get a key for each one so we can re-order
        # # shlist_len=len(tmp_edts)
        # # mykeys=np.arange(0,shlist_len,1)
        # #
        # # print mykeys
        #
        # edts_50 = tmp_edts[first50_ind]
        # cXs_50 = tmp_cXs[first50_ind]
        # cYs_50 = tmp_cYs[first50_ind]
        # degs_50 = tmp_degs[first50_ind]
        # mykeys_sel=mykeys[first50_ind]
        # chs_50 = tmp_chs[first50_ind]
        #
        # print mykeys_sel
        #
        # # Reorder
        # keyinds=np.argsort(mykeys_sel)
        # smp_edts = edts_50[keyinds]
        # smp_cXs = cXs_50[keyinds]
        # smp_cYs = cYs_50[keyinds]
        # smp_degs = degs_50[keyinds]
        # mykeys_final = mykeys_sel[keyinds]
        # smp_chs = chs_50[keyinds]
        #
        # print mykeys_final
        # print smp_edts
        #
        # # Find indices from var file
        # indices_m1 = []
        # for e in range(len(smp_edts)):
        #     date = smp_edts[e]
        #
        #     ix = my.ixdtimes(dtime, [date[0]], [date[1]], [date[2]], [0])
        #     if len(ix) >= 1:
        #         indices_m1.append(ix)
        #
        # indices_m1 = np.squeeze(np.asarray(indices_m1))
        #
        # # Select these dates
        # olrsel = olrdata[indices_m1, :, :]
        # dates = dtime[indices_m1]

        # Count timesteps
        nsteps=len(olrsel[:,0,0])

        ### Count number of events
        count_sel = str(nsteps)
        print "Total flagged CBs selected =" + str(count_sel)

        # Get lon lat grid
        plon, plat = np.meshgrid(lon, lat)

        # Loop 20 day intervals and plot
        tally=0
        nrep=2

        for r in range(nrep):
            print "repetition no "+str(r)
            fd=tally*nplot
            ld=fd+nplot
            thesedays=olrsel[fd:ld,:,:]
            datesel=smp_edts[fd:ld]
            cXsel=smp_cXs[fd:ld]
            cYsel=smp_cYs[fd:ld]
            degsel=smp_degs[fd:ld]
            chsel=smp_chs[fd:ld]

            #print datesel

            # Set up figure
            g, ax = plt.subplots(figsize=[12, 8])
            m, f = pt.AfrBasemap(lat, lon, drawstuff=True, prj='cyl', fno=1, rsltn='l')

            # Loop these 20 tsteps and make a plot
            cnt = 1
            for p in range(nplot):
                data4plot = np.squeeze(thesedays[p, :, :])
                tmp = datesel[p]
                datestr = str(tmp[0]) + '.' + str(tmp[1]) + '.' + str(tmp[2])
                this_cX=cXsel[p]
                this_cY=cYsel[p]
                this_deg=degsel[p]
                this_ch=chsel[p]

                # Plot subplot
                plt.subplot(yplots, xplots, cnt)
                clevs = np.arange(200, 280, 10)
                cm = plt.cm.gray_r
                cs = m.contourf(plon, plat, data4plot, clevs, cmap=cm, extend='both')


                # Add centroid
                #print 'Plotting centroid at lon'+str(this_cX)+' and lat '+str(this_cY)
                plt.plot(this_cX,this_cY,'o',c='fuchsia',zorder=1)

                # Draw contour angle
                ex = np.cos(np.deg2rad(this_deg)) * 6
                ey = np.sin(np.deg2rad(this_deg)) * 6
                #if this_deg < 0: ey = -ey
                #gpx = blb.Geopix((len(lat), len(lon)), lat, lon)
                #if hasattr(m, 'drawgreatcircle'):
                    #cx, cy = gpx.xp2lon(this_cX), gpx.yp2lat(cY)
                    #ex, ey = gpx.xp2lon(ex), gpx.yp2lat(ey)
                #    cx, cy = this_cX,this_cY
		        #    mcx, mcy = m(cx, cy)
                #    mex, mey = m(ex, ey)
                #    mex2, mey2 = m(-ex, -ey)
                #else:
                    #cx, cy = gpx.xp2lon(this_cX), gpx.yp2lat(this_cY)
                    #ex, ey = gpx.xp2lon(ex), gpx.yp2lat(ey)
                cx, cy = this_cX,this_cY
                mcx, mcy, mex, mey = cx, cy, ex, ey
                mex2, mey2 = -ex, -ey
                plt.arrow(mcx, mcy, mex, mey, fc='cyan', ec='r',zorder=2)
                #print 'Trying to plot arrow with...'
                #print mcx, mcy, mex, mey
                plt.arrow(mcx, mcy, mex2, mey2, fc='cyan', ec='r',zorder=3)
                txt = "Tilt: %03.0f" % (this_deg)
                plt.text(mcx, mcy, txt, color='c', fontsize=14., fontweight='bold')

                cnx, cny = this_ch[:, 0], this_ch[:, 1]
                plt.plot(cnx, cny, 'fuchsia', lw=1.)

                # Redraw map
                m.drawcountries()
                m.drawcoastlines()
                plt.title(datestr,fontsize='x-small')
                cnt += 1

            plt.subplots_adjust(left=0.05, right=0.9, top=0.98, bottom=0.02, wspace=0.1, hspace=0.2)

            # Plot cbar
            axcl = g.add_axes([0.95, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(cs, cax=axcl)


            # Save
            outname = outdir + 'looped_days_w_ch.'+str(tally)+'.n' + size + '.' + dset + '.' + name + '.' + globv + \
                      '.png'
            plt.savefig(outname, dpi=150)
            plt.close()

            tally+=1