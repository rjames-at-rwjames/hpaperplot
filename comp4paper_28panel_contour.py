# To plot all CMIP5 models in multi-panel plot
# using composites based on subselecting metbot flagged events by centroid and angle
# but now reading in netcdf files to speed things up

# Aiming at variables which I want to plot as a contour
# ctyp=abs
# ctyp=anom_mon - Composites plotted as anomaly from lt monthly mean (monthly mean for the month of each day selected)
# Now with option to add stippling if many days in composite agree

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import cm

cwd=os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd+'/../MetBot')
sys.path.append(cwd+'/../RTools')
import PlotTools as pt
import MetBot.dset_dict as dsetdict
import dsets_paper_28_4plot as dset_mp
import MetBot.mast_dset_dict as mast_dict
import MetBot.dimensions_dict as dim_exdict
import MetBot.mytools as my
import MetBot.mynetcdf as mync

### Running options
xplots = 4
yplots = 7
alphord=True
runs=['opt2']
ctyps=['abs'] #abs is absolute, anom is anomaly from ltmean, anom_mon is rt monthly mean
wcb=['cont'] # which cloud band composite? Options: cont, mada, dbl
spec_col=True
varlist=['olr']
thname='actual'
#levsel=True
levsel=False
if levsel:
    choosel=['500'] # can add a list
else:
    choosel=['1']
domain='swio'
if domain=='polar':
    sub='SH'
elif domain=='swio':
    sub='SA'
    figdim=[9,11]
elif domain=='nglob':
    sub='bigtrop'
elif domain=='mac_wave':
    sub='SASA'
    figdim=[9,9]
agtest=False # do a test on the proportion of days in comp that agree in dir change
#agtest=False # if abs then need to choose False
perc_ag=70 # show if this % or more days agree
#lag=True
lag=False
if lag:
    edays=[-3,-2,-1,0,1,2,3]
else:
    edays=[0]

#drawnest=True
drawnest=False
nestbox_cont=[23,35,-35,-20] # WESN
nestbox_mada=[35,50,-25,-11] # WESN

### Get directories
bkdir=cwd+"/../../CTdata/"
thisdir=bkdir+"/hpaperplot/"
botdir=bkdir+"metbot_multi_dset/"

compdir=thisdir+"comp4paper_contour/"
if lag:
    compdir=thisdir+"comp4paper_contour_lags/"
my.mkdir_p(compdir)

# variable
v=0
globv = varlist[v]
print "Running on "+globv

# levels - if levsel is false this will just be 1 level
l=0
print "Running on "+globv+" at pressure level "+choosel[l]


# Loop sampling options
for r in range(len(runs)):

    if runs[r]=='opt1':
        sample='blon'
        from_event='first'
    elif runs[r]=='opt2':
        sample='blon2'
        from_event='all'

    # Loop abs and anom
    for a in range(len(ctyps)):
        ctyp=ctyps[a]
        print "Running on "+ctyp

        # Loop sample type
        for o in range(len(wcb)):
            type = wcb[o]
            print "Running for sample " + type

            # Loop lags
            for lo in range(len(edays)):
                print "Running with a lag of "+str(edays[lo])

                # Set up plot
                print "Setting up plot..."
                g, ax = plt.subplots(figsize=figdim)

                cnt = 1

                ### Dsets
                # dsets='spec'
                # ndset=1
                # dsetnames=['noaa']
                dsets='all'
                ndset=len(dset_mp.dset_deets)
                #dsetnames=list(dset_mp.dset_deets)
                dsetnames=['noaa','cmip5']
                ndstr=str(ndset)

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
                        if lag:
                            smpfile = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/lag_samples/' \
                                      + name + '.' + name2 + '.' + globv + '.sampled_days.' \
                                      + sample + '.' + from_event + '.' + type + '.' + thname + '.lag_'+str(edays[lo])+'.nc'
                        else:
                            smpfile=bkdir+'metbot_multi_dset/'+dset2+'/'+name2+'/'\
                            +name+'.'+name2+'.'+globv+'.sampled_days.'\
                                +sample+'.'+from_event+'.'+type+'.'+thname+'.nc'

                        print 'Opening '+smpfile

                        if os.path.exists(smpfile):


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
                            nlat = len(lat)
                            nlon = len(lon)

                            if ctyp == 'anom_mon':

                                # Open ltmonmean file
                                meanfile = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
                                           + name2 + '.' + globv + '.mon.mean.' + ys + '.nc'

                                if os.path.exists(meanfile):

                                    print 'Opening ' + meanfile

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

                                else:
                                    print 'NO MEAN FILE AVAILABLE for ' + dset2 + '_' + name2
                                    print 'Setting mean to zero'
                                    meandata=np.zeros((12,nlat,nlon), dtype=np.float32)

                            if cnt == 1:
                            	m, f = pt.AfrBasemap(lat, lon, drawstuff=True, prj='cyl', fno=1, rsltn='l')

                            # Get lon lat grid
                            plon, plat = np.meshgrid(lon, lat)

                            # Get composite
                            print "Calculating composite..."
                            comp=np.nanmean(smpdata,0) # Check this is the right dimension to average ??
                            compdata = np.squeeze(comp)

                            if ctyp=='abs':

                                data4plot=compdata

                            elif ctyp=='anom_mon':
                                print "Calculating anomaly from long term MONTHLY means..."

                                anoms=np.zeros((nsamp,nlat,nlon),dtype=np.float32)
                                for day in range(nsamp):
                                    mon_thisday=smpdtime[day,1]
                                    this_monmean=meandata[mon_thisday-1]
                                    this_anom=smpdata[day,:,:]-this_monmean
                                    anoms[day,:,:]=this_anom

                                anom_comp=np.nanmean(anoms,0)
                                data4plot = anom_comp

                            if ctyp=='anom_mon':
                                if agtest:

                                    print "Calculating number of days which agree..."

                                    # Get signs
                                    anoms_signs = np.sign(anoms)
                                    comp_signs = np.sign(anom_comp)

                                    # Count ndays in composite with same sign as mean
                                    mask_zeros=np.zeros((nlat,nlon),dtype=np.float32)
                                    for i in range(nlat):
                                        for j in range(nlon):
                                            count=len(np.where(anoms_signs[:,i,j]==comp_signs[i,j])[0])
                                            perc=(float(count)/float(nsamp))*100
                                            if perc>=perc_ag:
                                                mask_zeros[i,j]=1
                                            else:
                                                mask_zeros[i,j]=0

                            # Plot
                            print "Plotting for model "+name2
                            plt.subplot(yplots,xplots,cnt)
                            if spec_col:
                                if globv == 'olr':
                                    if ctyp=='abs':
                                        clevs = np.arange(200, 280, 10)
                                        cm = plt.cm.gray_r
                                    elif ctyp=='anom_mon':
                                        #clevs = np.arange(-75,90,15)
                                        clevs= np.arange(-40,50,10)
                                        cm = plt.cm.BrBG_r
                                elif globv=='pr':
                                    if ctyp=='abs':
                                        clevs = np.arange(0, 16, 2)
                                        #cm = plt.cm.YlGnBu
                                        cm = plt.cm.magma
                                    elif ctyp=='anom_mon':
                                        clevs = np.arange(-12,14, 2)
                                        cm = plt.cm.bwr_r
                                elif globv == 'omega':
                                    clevs = np.arange(-0.12, 0.14, 0.02)
                                    cm = plt.cm.bwr
                                elif globv == 'q':
                                    clevs = np.arange(-0.004, 0.0045, 0.0005)
                                    cm = plt.cm.bwr_r
                                elif globv == 'gpth':
                                    clevs = np.arange(-60, 65, 5)
                                    cm = plt.cm.PiYG_r
                                cs = m.contourf(plon, plat, data4plot, clevs, cmap=cm, extend='both')
                            else:
                                cs = m.contourf(plon, plat, data4plot, extend='both')
                            if ctyp=='anom_mon':
                                if agtest:
                                    hatch = m.contourf(plon, plat, mask_zeros, levels=[-1.0, 0.0, 1.0], hatches=["", '.'], alpha=0)

                            plt.title(name2,fontsize=8,fontweight='demibold')

                            # Redraw map
                            m.drawcountries()
                            m.drawcoastlines()

                            # Draw box
                            if drawnest:
                                if type=='cont':
                                    nestbox=nestbox_cont
                                elif type=='mada':
                                    nestbox=nestbox_mada

                                nest_x=[nestbox[0],nestbox[0],nestbox[1],nestbox[1],nestbox[0]]
                                nest_y=[nestbox[2],nestbox[3],nestbox[3],nestbox[2],nestbox[2]]
                                plt.plot(nest_x,nest_y,color='k',linewidth=2)

                            cnt += 1


                        else:
                            print 'NO sample FILE AVAILABLE for ' + dset2 + '_' + name2
                            print 'Moving to next model....'
                            cnt += 1

                print "Finalising plot..."
                plt.subplots_adjust(left=0.05,right=0.9,top=0.95,bottom=0.02,wspace=0.1,hspace=0.2)

                # Plot cbar
                axcl = g.add_axes([0.91, 0.15, 0.01, 0.6])
                cbar = plt.colorbar(cs, cax=axcl)
                my.ytickfonts(fontsize=12.)

                # Save
                if ctyp == 'anom_mon':
                    if agtest:
                        cstr=ctyp+'_agtest_'+str(perc_ag)
                    else:
                        cstr = ctyp
                else:
                    cstr=ctyp
                if lag:
                    compname = compdir + 'multi_comp_' + cstr + '.' + sample + '.' + type + '.' + globv + \
                               '.' + choosel[l] + '.' + sub + '.from_event' + from_event + '.lag_'+str(edays[lo])+'.png'
                else:
                    compname = compdir + 'multi_comp_'+cstr+'.'+sample+'.' + type + '.' + globv + \
                          '.'+choosel[l]+'.'+sub+'.from_event'+from_event+'.png'
                print 'saving figure as '+compname
                plt.savefig(compname, dpi=150)
                plt.close()
