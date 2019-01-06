# To show variation within model composites
# by plotting the outline of the CB
# this is also a check that I am selecting them correctly in ANOVA analysis
# and potentially a way to figure out at which latitudes or longitudes there might be more or less similarity


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
cb_test=False # start by just plotting one CB

plotshow='col5' # 'greyall' or 'col5'

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


                # Make a map
                plt.subplot(yplots, xplots, cnt)
                print 'Generating map'
                m, mfig = pt.AfrBasemap(lat, lon, drawstuff=True, prj='cyl', fno=1, rsltn='l')

                nch=len(chs_smpl)
                if cb_test:
                    nch=1
                if plotshow=='col5':
                    nch=5
                    cols = ['r', 'b', 'c', 'm', 'g', 'r', 'b', 'c', 'm', 'g']

                print 'Looping sample days and plotting CB outlines'
                for jl in range(nch):
                    cb=chs_smpl[jl]
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
            end=+'_test.'

        compname= plotdir + 'cb_outlines_plot.'+plotshow+'.sample_'+type+'.'+globv+'.models_'+mods+'.'+end+'png'
        plt.savefig(compname, dpi=150)
        plt.close()