# Plotting wrapper
# to plot
# ....precip associated with TTTs
# ....in a grid with 28 CMIP models
# ....for the season NDJFM
#
# OLR threshold is detected automatically using "find_saddle"
# Option to run on other OLR thresholds a test - currently + and - 5Wm2
#
#
# .....directory: here ../../CTdata/metbot_multi_dset/$dset/
# naming of ncfiles used here /$dset/$name.olr.day.mean.$firstyear_$lastyear.nc

import os
import sys
from datetime import date

import matplotlib.pyplot as plt
import numpy as np

cwd=os.getcwd()
sys.path.append(cwd+'/..')
sys.path.append(cwd+'/../RTools')
sys.path.append(cwd+'/../MetBot')
import MetBot.SynopticAnatomy as sy
import MetBot.EventStats as stats
import MetBot.AdvancedPlots as ap
import MetBot.MetBlobs as blb
import MetBot.mytools as my
import MetBot.mynetcdf as mync
import MetBot.dset_dict as dsetdict
import dsets_mplot_5group_4plot as dset_mp

### Running options
sub="SA"
subrain="SA_TR"
#subrain="SA_CONT"
#subrain="UM_FOC"
print 'Mapping for domain '+subrain
seasons=['NDJFM']
thname='upper'         # Option to run on thresholds + and - 5Wm2 as a test
xplots=4
yplots=7
group=True


### Plot type
metatype='ttt' # 'all' or 'ttt' - is it a plot for all rain or just TTT rain
plottype='rain_per_ttt'
heavy=False
perc_ag=70              # show if this % or more days agree
print 'Running for plottype '+plottype


### Plot types

## metatype all - heavy=False

# options to plot all days
# 'tot_all'            # plot total rainfall
# 'all_cnt'       # plot % of days which have +ve anomalies

## metatype all - heavy=True

# options to plot all - heavy pr
# 'all_wet_cnt'       # plot number of days over hvthr - either total or per mon depending on 'monmean'
# 'all_wet_sum'       # plot rainfall from days over hvthr - either total or per mon depending on 'monmean'
# 'aper_wet_cnt'       # plot % of days that are over hvthr
# 'aper_wet_sum'       # plot % of precip which is falling on days over hvthr

## metatype ttt - heavy=False

# options to plot ttt...
# 'tot_ttt'      # plot total rainfall from TTTs
# 'per_ttt'      # plot percentage rainfall from TTTs (tot_ttt/tot_all)
# 'rain_per_ttt'  # plot average rain per TTT day (rain composite)
# 'comp_anom_ttt'  # plot rain per TTT as anom from long term daily mean for each month
# 'comp_anom_ag' # plot comp anom with agtest on composite
# 'comp_anom_cnt'     # plot count of the number of days above or below average

## metatype ttt - heavy=True

# options to plot ttt - heavy pr
# 'ttt_wet_cnt'        # plot number of  days over hvthr - either total or per mon depending on 'monmean'
# 'tper_wet_cnt'       # % of days over hvthr contributed by TTTs
# 'per_tttd_wet'      # plot % of TTT days which have precip over this threshold
# 'ttt_wet_sum'       # plot rainfall from days over hvthr - either total or per mon depending on 'monmean'
# 'tper_wet_sum'       # % of precip from days over hvthr contributed by TTTs

under_dayof='under'     # if "dayof" plots all rain on TTT days
                        #   if "under" plots rain under TTTs (based on blobs)

monmean='day'           # to control the output - is there averaging?
                        # 'day' is daily mean - note that day is not currently
                        #          set up to work with all opts e.g. wet day counts
                        # 'mon' is monthly mean
                        # 'tot' is total
nTTTlab=False            # labels each plot with # or % of TTTs

freecol=False           # free colour bar
refkey='0'              # 0 or all

if metatype=='all':
    doms=['All']
elif metatype=='ttt':
    # doms=['All']
    doms=['All','nCont','nMada','nOcea'] # doms for TTT days selected

if heavy:
    hvthrs=['0.5','10','25','50']
else:
    hvthrs=['0.5']


if subrain=='SA_TRMM':
    figdim=[9,11]
elif subrain=='SA_CONT':
    figdim=[9,11]
elif subrain=='UM_FOC':
    figdim=[9,11]
elif subrain=='SA_TR':
    figdim=[10,7]

if group:
    grcls=['fuchsia','gold','darkblue','r','blueviolet','springgreen']


### Get directories
bkdir=cwd+"/../../CTdata/"
botdir=bkdir+"metbot_multi_dset/"
thisdir=bkdir+"hpaperplot/"
prdir=thisdir+"precip_figs_allCMIP/"
my.mkdir_p(prdir)

### If ttt plot loop domains
for do in range(len(doms)):
    dom=doms[do]
    print 'Running on domain '+dom

    ### If heavy plot loop heavythres
    for he in range(len(hvthrs)):
        hvthr=hvthrs[he]
        print 'Running on heavythres '+hvthr

        for s in range(len(seasons)):

            seas = seasons[s]
            print 'Running for season '+seas

            if seas == 'NDJFM':
                mon1 = 11
                mon2 = 3
                nmon = 5

            if seas == 'DJF':
                mon1 = 12
                mon2 = 2
                nmon = 3

            if seas == 'DJ':
                mon1 = 12
                mon2 = 1
                nmon = 2

            # Set up plot
            print "Setting up plot..."
            g,ax = plt.subplots(figsize=figdim)
            cnt = 1
            ### Finalising plot
            if metatype=='all':
                mapsuf = plottype+'_'+seas + '_' + subrain + '_4'+ monmean
            elif metatype=='ttt':
                mapsuf = plottype+'_'+seas + '_' + subrain + '_4'+ monmean +'_'+thname+'_'+under_dayof+'_'+dom

            if heavy:
                mapsuf=mapsuf+'_hvthr'+hvthr

            if plottype=='comp_anom_ag':
                mapsuf=mapsuf+'_perc_ag'+str(perc_ag)

            if group:
                mapsuf=mapsuf+'_grouped.'

            ### Multi dset?
            # dsets='spec'
            # ndset=1
            # dsetnames=['noaa']
            dsets='all'
            ndset = len(dset_mp.dset_deets)
            dsetnames = ['noaa', 'cmip5']
            ndstr = str(ndset)

            print "Looping datasets"
            for d in range(ndset):
                dset=dsetnames[d]
                dcnt=str(d+1)
                print 'Running on '+dset
                print 'This is dset '+dcnt+' of '+ndstr+' in list'

                ### Multi model?
                mods='all'
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

                    else:
                        mnames = mnames_tmp
                else:
                    mnames = mnames_tmp

                for mo in range(nmod):
                    name=mnames[mo]
                    mcnt=str(mo+1)
                    print 'Running on ' + name
                    print 'This is model '+mcnt+' of '+nmstr+' in list'

                    # Get details
                    moddct=dsetdict.dset_deets[dset][name]

                    if group:
                        groupdct = dset_mp.dset_deets[dset][name]
                        thisgroup = int(groupdct['group'])
                        grcl = grcls[thisgroup - 1]

                    ### Location for input & outputs
                    indir=botdir+dset+"/"
                    outdir=indir+name+"/"
                    outsuf=outdir+name+'_'
                    cal = moddct['calendar']
                    ys = moddct['yrfname']
                    beginatyr = moddct['startyr']

                    ### Open rain data
                    globp = 'pr'
                    if dset == 'noaa':
                        raindset = 'trmm'
                        rainmod = 'trmm_3b42v7'
                        rmoddct = dsetdict.dset_deets[raindset][rainmod]
                        rcal = rmoddct['calendar']
                        rys = rmoddct['yrfname']
                        rbeginatyr = rmoddct['startyr']
                    else:
                        raindset = dset
                        rainmod = name
                        rmoddct = moddct
                        rcal = cal
                        rys = ys
                        rbeginatyr = beginatyr

                    rainname = rmoddct['prname']
                    rainfile = botdir + raindset + "/" + rainmod + "."+globp+".day.mean." + rys + ".nc"
                    print 'Opening '+rainfile

                    rainout = mync.open_multi(rainfile, globp, rainmod, \
                                              dataset=raindset, subs=subrain)

                    rdim = len(rainout)
                    if rdim == 5:
                        rain, rtime, rlat, rlon, rdtime = rainout
                    elif rdim == 6:
                        rain, rtime, rlat, rlon, rlev, rdtime = rainout
                        rain = np.squeeze(rain)
                    else:
                        print 'Check number of levels in ncfile'
                    rdtime[:, 3] = 0

                    print 'Checking for duplicate timesteps' # do retain this - IPSL A LR has double tsteps
                    tmp = np.ascontiguousarray(rdtime).view(np.dtype((np.void, rdtime.dtype.itemsize * rdtime.shape[1])))
                    _, idx = np.unique(tmp, return_index=True)
                    rdtime = rdtime[idx]
                    rain = rain[idx, :, :]

                    ### Select data to run
                    start = rmoddct['startdate']
                    ystart = int(start[0:4]);
                    mstart = int(start[5:7]);
                    dstart = int(start[8:10])
                    if rcal == "360_day":
                        startday = (ystart * 360) + ((mstart - 1) * 30) + dstart
                        beginday = ((int(rbeginatyr)) * 360) + 1
                        daysgap = beginday - startday + 1
                    else:
                        startd = date(ystart, mstart, dstart)
                        begind = date(int(rbeginatyr), 01, 01)
                        daysgap = (begind - startd).days
                    rain = rain[daysgap:, :, :];
                    rtime = rtime[daysgap:];
                    rdtime = rdtime[daysgap:]

                    if metatype=='ttt':
                        print 'Getting TTT data...'

                        ### Get threshold
                        print 'getting threshold....'
                        threshtxt=botdir+'thresholds.fmin.all_dset.txt'
                        with open(threshtxt) as f:
                            for line in f:
                                if dset+'\t'+name in line:
                                    thresh = line.split()[2]
                                    print 'thresh='+str(thresh)
                        thresh = int(thresh)
                        if thname=='actual':
                            thisthresh=thresh
                        elif thname=='lower':
                            thisthresh=thresh - 5
                        elif thname=='upper':
                            thisthresh = thresh + 5
                        print thisthresh
                        thre_str=str(int(thisthresh))

                        mbsfile=outsuf+thre_str+'_'+dset+"-olr-0-0.mbs"
                        syfile=outsuf+thre_str+'_'+dset+'-OLR.synop'

                        ### Open ttt data
                        print 'opening metbot files...'
                        s = sy.SynopticEvents((),[syfile],COL=False)
                        refmbs, refmbt, refch = blb.mbopen(mbsfile)

                        ### Get all events
                        ks = s.events.keys();ks.sort() # all
                        count_all=str(int(len(ks)))
                        print "Total CB events ="+str(count_all)
                        key = dset + '-olr-0-' + refkey

                        ### Select the season
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
                            if (int(dt[1]) >= mon1) or (int(dt[1]) <= mon2):
                                thesekeys.append(k)
                                edts.append(dt)
                        edts = np.asarray(edts)
                        yrs = np.unique(edts[:, 0])

                        # Split by domain
                        print 'selecting events for this dom group...'
                        if len(doms) == 1:
                            keys = [thesekeys]
                        elif len(doms) == 4:
                            k1, ktmp = stats.spatialsubset(s, thesekeys, cutlon=37.5)
                            k2, k3 = stats.spatialsubset(s,ktmp,cutlon=67.5)
                            if dom =='All':
                                keys=ks
                            elif dom=='nCont':
                                keys=k1
                            elif dom=='nMada':
                                keys=k2
                            elif dom=='nOcea':
                                keys=k3

                    else:
                        s='synop not needed'
                        keys='keys not needed'
                        key='key not needed'


                    print 'Plotting for model '+rainmod
                    plt.subplot(yplots,xplots,cnt)
                    allmask=ap.gridrainmap_single(s,keys,rain,rlat,rlon,rdtime,rainmod,\
                                                  season=seas,key=key,ptype=plottype,mmean=monmean,\
                                                  under_of=under_dayof, \
                                                  savefig=False, labels=nTTTlab,\
                                                  agthresh=perc_ag, heavy=hvthr,bound=grcl)

                    cnt +=1


            ### Finalising plot
            print 'Finalising plot'
            plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.02, wspace=0.1, hspace=0.2)
            figname=prdir+'Rainmap_'+mapsuf+'.png'
            print 'Saving figure as '+figname
            plt.savefig(figname, dpi=150)
            plt.close()
