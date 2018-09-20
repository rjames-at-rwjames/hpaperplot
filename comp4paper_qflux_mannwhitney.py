# To plot all CMIP5 models in multi-panel plot
# This one is for qflux and q at 850
# so only 27 mods because of missing data
# but now using composites based on subselecting metbot flagged events by centroid and angle
# but now reading in netcdf files to speed things up
#
#
# Aiming at variables which I want to plot as a vector
# ctyp=abs
# ctyp=anom_mon - Composites plotted as anomaly from lt monthly mean (monthly mean for the month of each day selected)
# with option to only plot vectors if they are shown by specified % of days in composites

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi
from mpl_toolkits.basemap import cm

cwd=os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd+'/../MetBot')
sys.path.append(cwd+'/../RTools')
import dsets_paper_qflux as dset_mp
import MetBot.dset_dict as dsetdict
import MetBot.mast_dset_dict as mast_dict
import MetBot.dimensions_dict as dim_exdict
import MetBot.mytools as my
import MetBot.mynetcdf as mync
import PlotTools as pt
import scipy


### Running options
test_scr=False
xplots = 4
yplots = 7
runs=['opt3']
wcb=['cont'] # which cloud band composite? Options: cont, mada, dbl
spec_col=True
domain='swio' # swio or mac_wave
agtest=False # do a test on the proportion of days in comp that agree in dir change
perc_ag=75 # show if this % or more days agree

lag=False

thname='actual'
rean='era' # reanalysis - ncep or era
alphord=True
interp=True
int_res=2.5

manntest=True
fdr=True
alphaFDR=0.05

seas='NDJFM'
climyr='spec' # this is to use new climatology files which are based on only 35 years
                # 'spec' is 35 years
                # 'prev' is previous files - all different climatologies
# Info for vector
varlist=['qflux']
ctyp='anom_mon' #abs is absolute,  anom_mon is rt monthly mean, anom_seas is rt seasonal mean
levsel=True
if levsel:
    choosel=['850'] # can add a list
else:
    choosel=['1']


if ctyp=='abs':
    skip=2
elif ctyp=='anom_mon' or ctyp=='anom_seas':
    skip=2 # to skip over some vectors in plotting. If want to skip none use 1

# Info for contour
pluscon=True
convar=['q']
ctyp_con='anom_mon'
levcon=True
if levcon:
    chooselc=['850'] # can add a list
else:
    chooselc=['1']
agtest_con=False
perc_ag_con=75

# Lag info
if lag:
    edays=[-3,-2,-1,0,1,2,3] # lags only currently working with u and v
else:
    edays=[0]

# Info on domains
if domain=='polar':
    sub='SH'
elif domain=='swio':
    sub='SA'
    figdim=[9,11]
elif domain=='nglob':
    sub='bigtrop'
elif domain=='mac_wave':
    sub='SASA'
    figdim = [9, 7]

### Get directories
bkdir=cwd+"/../../CTdata/"
thisdir=bkdir+"/hpaperplot/"
botdir=bkdir+"metbot_multi_dset/"

figdir=thisdir+"comp4paper_qflux/"
if lag:
    figdir=thisdir+"comp4paper_qflux_lags/"
my.mkdir_p(figdir)

if seas == 'NDJFM':
    mons=[1,2,3,11,12]
    mon1 = 11
    mon2 = 3
    nmon = 5

# vector variable
v=0
variable = varlist[v]
print "Running on " + variable
if variable == 'wind':
    globv1 = 'u'
    globv2 = 'v'
elif variable == 'qflux':
    globv1 = 'u'
    globv2 = 'v'
    globv3 = 'q'

if pluscon:
    globv_c = convar[v]

# levels - if levsel is false this will just be 1 level
l=0
print "Running on " + variable + " at pressure level " + choosel[l]

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
        print "Running for sample " + type

        # Loop lags
        for lo in range(len(edays)):
            print "Running with a lag of "+str(edays[lo])

            # Set up plot
            print "Setting up plot..."
            g, ax = plt.subplots(figsize=figdim)

            cnt = 1

            ### Dsets
            dsets='all'
            dsetnames=['noaa','cmip5']
            ndset=len(dsetnames)
            ndstr=str(ndset)

            if test_scr:
                ndset = 1

            print "Looping datasets"
            for d in range(ndset):
                dset=dsetnames[d]
                dcnt=str(d+1)
                print 'Running on '+dset
                print 'This is dset '+dcnt+' of '+ndstr+' in list'

                if dset != 'cmip5': levc = int(choosel[l])
                else: levc = int(choosel[l]) * 100

                if pluscon:
                    if dset != 'cmip5':
                        lev_c = int(chooselc[l])
                    else:
                        lev_c = int(chooselc[l]) * 100

                ### Models
                mods = 'all'
                nmod = len(dset_mp.dset_deets[dset])
                mnames_tmp = list(dset_mp.dset_deets[dset])
                nmstr = str(nmod)

                if dset == 'cmip5':
                    if alphord:
                        mnames = sorted(mnames_tmp, key=lambda s: s.lower())
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

                    # Switch variable if NOAA
                    if dset == 'noaa':
                        if rean=='ncep':
                            ds4noaa = 'ncep'
                            mod4noaa = 'ncep2'
                        elif rean=='era':
                            ds4noaa = 'era'
                            mod4noaa = 'erai'
                        dset2 = ds4noaa
                        name2 = mod4noaa
                    else:
                        dset2 = dset
                        name2 = name

                    if pluscon:
                        if dset=='noaa':
                            if globv_c == 'olr':
                                dset3 = dset
                                name3 = name
                            else:
                                if rean == 'ncep':
                                    ds4noaa = 'ncep'
                                    mod4noaa = 'ncep2'
                                elif rean == 'era':
                                    ds4noaa = 'era'
                                    mod4noaa = 'erai'
                                dset3 = ds4noaa
                                name3 = mod4noaa
                        else:
                            dset3 = dset
                            name3 = name

                    # Get info
                    moddct = dsetdict.dset_deets[dset2][name2]
                    mastdct = mast_dict.mast_dset_deets[dset2]

                    if pluscon:
                        condct = dsetdict.dset_deets[dset3][name3]
                        conmastdct=mast_dict.mast_dset_deets[dset3]

                    vnamedict_u = globv1 + 'name'
                    vnamedict_v = globv2 + 'name'
                    if variable == 'qflux':
                        vnamedict_q = globv3 + 'name'

                    if pluscon:
                        vnamedict_c = globv_c + 'name'


                    varstr_u = mastdct[vnamedict_u]
                    varstr_v = mastdct[vnamedict_v]
                    if variable == 'qflux':
                        varstr_q = mastdct[vnamedict_q]

                    if pluscon:
                        varstr_c = conmastdct[vnamedict_c]

                    ys=moddct['fullrun']
                    if pluscon:
                        if globv_c != 'omega' and globv_c != 'q' and globv_c != 'gpth':
                            ys_c = condct['yrfname']
                        else:
                            if name3 == "MIROC5":
                                if globv_c == 'q':
                                    ys_c = condct['fullrun']
                                elif globv_c == 'omega' or globv_c == 'gpth':
                                    ys_c = '1950_2009'
                                else:
                                    print 'variable ' + globv_c + ' has unclear yearname for ' + name3
                            else:
                                ys_c = condct['fullrun']


                    # Years for clim and manntest
                    if climyr == 'spec':
                        ysclim = moddct['yrfname']
                    else:
                        ysclim = ys
                    year1 = float(ysclim[0:4])
                    year2 = float(ysclim[5:9])

                    if pluscon:
                        if climyr == 'spec':
                            ysclim_c = condct['yrfname']
                        else:
                            ysclim_c = ys_c
                        year1_c = float(ysclim_c[0:4])
                        year2_c = float(ysclim_c[5:9])


                    dimdict = dim_exdict.dim_deets[globv1][dset2]
                    latname = dimdict[1]
                    lonname = dimdict[2]

                    if pluscon:
                        dimdict_c = dim_exdict.dim_deets[globv_c][dset3]
                        latname_c = dimdict_c[1]
                        lonname_c = dimdict_c[2]

                    # Open sample files
                    if lag:
                        smpfile_u = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/lag_samples/' \
                                    + name + '.' + name2 + '.' + globv1 + '.sampled_days.' \
                                    + sample + '.' + from_event + '.' + type + '.' + thname + '.lag_'+str(edays[lo])+'.nc'
                        smpfile_v = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/lag_samples/' \
                                    + name + '.' + name2 + '.' + globv2 + '.sampled_days.' \
                                    + sample + '.' + from_event + '.' + type + '.' + thname + '.lag_'+str(edays[lo])+'.nc'
                    else:
                        smpfile_u = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 +'/' \
                                    +name+'.'+name2+'.' + globv1 + '.sampled_days.'\
                                    + sample + '.'+from_event+'.'+type+'.'+thname+'.nc'
                        smpfile_v = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 +'/' \
                                    +name+'.'+name2+'.' + globv2 + '.sampled_days.'\
                                    + sample + '.'+from_event+'.'+type+'.'+thname+'.nc'

                    if variable == 'qflux':
                        if lag:
                            smpfile_q = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/lag_samples/' \
                                        + name + '.' + name2 + '.' + globv3 + '.sampled_days.' \
                                        + sample + '.' + from_event + '.' + type + '.' + thname + '.lag_'+str(edays[lo])+'.nc'
                        else:
                            smpfile_q = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 +'/' \
                                +name+'.'+name2+'.' + globv3 + '.sampled_days.'\
                                + sample + '.'+from_event+'.'+type+'.'+thname+'.nc'
                    if pluscon:
                        if lag:
                            smpfile_c = bkdir + 'metbot_multi_dset/' + dset3 + '/' + name3 + '/lag_samples/' \
                                        + name + '.' + name3 + '.' + globv_c + '.sampled_days.' \
                                        + sample + '.' + from_event + '.' + type + '.' + thname + '.lag_'+str(edays[lo])+'.nc'
                        else:
                            smpfile_c = bkdir + 'metbot_multi_dset/' + dset3 + '/' + name3 +'/' \
                                +name+'.'+name3+'.' + globv_c + '.sampled_days.'\
                                + sample + '.'+from_event+'.'+type+'.'+thname+'.nc'

                    if manntest:
                        # Open all file
                        allfile_u = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + \
                                      '.' + globv1 + '.day.mean.' + ys + '.nc'

                        allfile_v = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + \
                                      '.' + globv2 + '.day.mean.' + ys + '.nc'

                        if variable == 'qflux':
                            allfile_q = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + \
                                        '.' + globv3 + '.day.mean.' + ys + '.nc'

                        if pluscon:
                            allfile_c = bkdir + 'metbot_multi_dset/' + dset3 + '/' + name3 + \
                                        '.' + globv_c + '.day.mean.' + ys_c + '.nc'

                    print 'Opening ' + smpfile_u
                    print 'and corresponding file: ' + smpfile_v
                    if variable == 'qflux':
                        print 'and q file:' + smpfile_q
                    if pluscon:
                        print 'and file for '+globv_c +' ' +smpfile_c
                    if manntest:
                        print 'and files for whole period:'
                        print allfile_u
                        print allfile_v
                        if variable =='qflux':
                            print allfile_q
                        if pluscon:
                            print allfile_c

                    if levsel:
                        ncout_u = mync.open_multi(smpfile_u, globv1, name2, \
                                                dataset=dset2, subs=sub, levsel=levc)
                        ncout_v = mync.open_multi(smpfile_v, globv2, name2, \
                                                dataset=dset2, subs=sub, levsel=levc)

                        if variable == 'qflux':

                            ncout_q = mync.open_multi(smpfile_q, globv3, name2, \
                                                      dataset=dset2, subs=sub, levsel=levc)

                    else:
                        ncout_u = mync.open_multi(smpfile_u, globv1, name2, \
                                                dataset=dset2, subs=sub)
                        ncout_v = mync.open_multi(smpfile_v, globv2, name2, \
                                                dataset=dset2, subs=sub)

                        if variable == 'qflux':
                            ncout_q = mync.open_multi(smpfile_q, globv3, name2, \
                                                      dataset=dset2, subs=sub)

                    if pluscon:
                        if levcon:
                            ncout_c = mync.open_multi(smpfile_c, globv_c, name3, \
                                                  dataset=dset3, subs=sub,levsel=lev_c)
                        else:
                            ncout_c = mync.open_multi(smpfile_c, globv_c, name3, \
                                                  dataset=dset3, subs=sub)

                    ndim = len(ncout_u)
                    if pluscon:
                        ndim_c = len(ncout_c)

                    if ndim == 5:

                        smpdata_u, time, lat, lon, smpdtime = ncout_u
                        smpdata_v, time, lat, lon, smpdtime = ncout_v

                        if variable == 'qflux':
                            smpdata_q, time, lat, lon, smpdtime = ncout_q

                    elif ndim == 6:

                        smpdata_u, time, lat, lon, lev, smpdtime = ncout_u
                        smpdata_u = np.squeeze(smpdata_u)

                        smpdata_v, time, lat, lon, lev, smpdtime = ncout_v
                        smpdata_v = np.squeeze(smpdata_v)

                        if variable == 'qflux':
                            smpdata_q, time, lat, lon, lev, smpdtime = ncout_q
                            smpdata_q = np.squeeze(smpdata_q)

                    else:
                        print 'Check number of dims in ncfile'

                    smpdtime[:, 3] = 0

                    if pluscon:

                        if ndim_c == 5:

                            smpdata_c, time_c, lat_c, lon_c, smpdtime_c = ncout_c

                        elif ndim_c == 6:

                            smpdata_c, time_c, lat_c, lon_c, lev_c, smpdtime_c = ncout_c
                            smpdata_c = np.squeeze(smpdata_c)

                        smpdtime_c[:, 3] = 0

                    # Fix lat and lons if it spans 0
                    if domain == 'mac_wave' or domain == 'bigtrop':
                        print "Ammending lons around 0"
                        for i in range(len(lon)):
                            if lon[i] > 180:
                                lon[i] = lon[i] - 360
                        ord = np.argsort(lon)
                        lon = lon[ord]
                        smpdata_u = smpdata_u[:, :, ord]
                        smpdata_v = smpdata_v[:, :, ord]

                        if variable=='qflux':
                            smpdata_q = smpdata_q[:,:, ord]

                        if pluscon:
                            for i in range(len(lon_c)):
                                if lon_c[i] > 180:
                                    lon_c[i] = lon_c[i] - 360
                            ord = np.argsort(lon_c)
                            lon_c = lon_c[ord]
                            smpdata_c = smpdata_c[:, :, ord]

                    nsamp = len(smpdata_u[:, 0, 0])
                    nlat = len(lat)
                    nlon = len(lon)


                    if manntest:

                        # Open all file
                        if levsel:
                            ncout_u = mync.open_multi(allfile_u, globv1, name2, \
                                                    dataset=dset2, subs=sub, levsel=levc)

                            ncout_v = mync.open_multi(allfile_v, globv2, name2, \
                                                      dataset=dset2, subs=sub, levsel=levc)

                            if variable == 'qflux':
                                ncout_q = mync.open_multi(allfile_q, globv3, name2, \
                                                          dataset=dset2, subs=sub, levsel=levc)

                        else:
                            ncout_u = mync.open_multi(allfile_u, globv1, name2, \
                                                    dataset=dset2, subs=sub)

                            ncout_v = mync.open_multi(allfile_v, globv2, name2, \
                                                dataset=dset2, subs=sub)

                            if variable == 'qflux':
                                ncout_q = mync.open_multi(allfile_q, globv3, name2, \
                                                          dataset=dset2, subs=sub)

                        if pluscon:
                            if levcon:
                                ncout_c = mync.open_multi(allfile_c, globv_c, name3, \
                                                          dataset=dset3, subs=sub, levsel=lev_c)
                            else:
                                ncout_c = mync.open_multi(allfile_c, globv_c, name3, \
                                                          dataset=dset3, subs=sub)


                        print '...files opened'
                        ndim = len(ncout_u)
                        if ndim == 5:
                            alldata_u, time, lat, lon, alldtime = ncout_u
                            alldata_v, time, lat, lon, alldtime = ncout_v

                            if variable == 'qflux':
                                alldata_q, time, lat, lon, alldtime = ncout_q

                        elif ndim == 6:
                            alldata_u, time, lat, lon, lev, alldtime = ncout_u
                            alldata_u = np.squeeze(alldata_u)

                            alldata_v, time, lat, lon, lev, alldtime = ncout_v
                            alldata_v = np.squeeze(alldata_v)

                            if variable == 'qflux':
                                alldata_q, time, lat, lon, lev, alldtime = ncout_q
                                alldata_q = np.squeeze(alldata_q)
                        else:
                            print 'Check number of dims in ncfile'
                        alldtime[:, 3] = 0

                        if pluscon:
                            ndim_c = len(ncout_c)

                            if ndim_c == 5:

                                alldata_c, time_c, lat_c, lon_c, alldtime_c = ncout_c

                            elif ndim_c == 6:

                                alldata_c, time_c, lat_c, lon_c, lev_c, alldtime_c = ncout_c
                                alldata_c = np.squeeze(alldata_c)

                            alldtime_c[:, 3] = 0

                        # Fix lat and lons if it spans 0
                        if domain == 'mac_wave' or domain == 'bigtrop':
                            print "Ammending lons around 0"
                            for i in range(len(lon)):
                                if lon[i] > 180:
                                    lon[i] = lon[i] - 360
                            ord = np.argsort(lon)
                            lon = lon[ord]
                            alldata_u = alldata_u[:, :, ord]
                            alldata_v = alldata_v[:, :, ord]

                            if variable == 'qflux':
                                alldata_q = alldata_q[:, :, ord]

                            if pluscon:
                                for i in range(len(lon_c)):
                                    if lon_c[i] > 180:
                                        lon_c[i] = lon_c[i] - 360
                                ord = np.argsort(lon_c)
                                lon_c = lon_c[ord]
                                alldata_c = alldata_c[:, :, ord]

                        # Remove duplicate timesteps
                        print 'Checking for duplicate timesteps'
                        tmp = np.ascontiguousarray(alldtime).view(np.dtype((np.void, alldtime.dtype.itemsize * alldtime.shape[1])))
                        _, idx = np.unique(tmp, return_index=True)
                        alldtime = alldtime[idx]
                        alldata_u = alldata_u[idx, :, :]
                        alldata_v = alldata_v[idx, :, :]
                        if variable == 'qflux':
                            alldata_q = alldata_q[idx, :, :]

                        if pluscon:
                            print 'Checking for duplicate timesteps'
                            tmp = np.ascontiguousarray(alldtime_c).view(
                                np.dtype((np.void, alldtime_c.dtype.itemsize * alldtime_c.shape[1])))
                            _, idx = np.unique(tmp, return_index=True)
                            alldtime_c = alldtime_c[idx]
                            alldata_c = alldata_c[idx, :, :]

                        nsteps = len(alldtime)
                        if pluscon:
                            nsteps_c = len(alldtime_c)

                        sinds = []
                        odts = []
                        oinds = []
                        for dt in range(nsteps):
                            thisdate = alldtime[dt]
                            if thisdate[0] >= year1 and thisdate[0] <= year2:
                                if thisdate[1] >= mon1 or thisdate[1] <= mon2:
                                    ix = my.ixdtimes(smpdtime, [thisdate[0]], \
                                                     [thisdate[1]], [thisdate[2]], [0])
                                    if len(ix) == 1:
                                        sinds.append(dt)
                                    elif len(ix) < 1:
                                        odts.append(thisdate)
                                        oinds.append(dt)

                        odts = np.asarray(odts)
                        oinds = np.asarray(oinds)
                        sinds = np.asarray(sinds)

                        if pluscon:
                            sinds_c = []
                            odts_c = []
                            oinds_c = []
                            for dt in range(nsteps_c):
                                thisdate = alldtime_c[dt]
                                if thisdate[0] >= year1_c and thisdate[0] <= year2_c:
                                    if thisdate[1] >= mon1 or thisdate[1] <= mon2:
                                        ix = my.ixdtimes(smpdtime_c, [thisdate[0]], \
                                                         [thisdate[1]], [thisdate[2]], [0])
                                        if len(ix) == 1:
                                            sinds_c.append(dt)
                                        elif len(ix) < 1:
                                            odts_c.append(thisdate)
                                            oinds_c.append(dt)

                            odts_c = np.asarray(odts_c)
                            oinds_c = np.asarray(oinds_c)
                            sinds_c = np.asarray(sinds_c)

                    # If anom open longterm mean files
                    if ctyp == 'anom_mon' or ctyp == 'anom_seas':

                        meanfile_u=bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
                                       + name2 + '.' + globv1 + '.mon.mean.' + ysclim + '.nc'

                        meanfile_v=bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
                                       + name2 + '.' + globv2 + '.mon.mean.' + ysclim + '.nc'

                        if variable == 'qflux':
                            meanfile_q = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
                                         + name2 + '.' + globv3 + '.mon.mean.' + ysclim + '.nc'



                        print 'Opening ' + meanfile_u
                        print 'and corresponding file: ' + meanfile_v
                        if variable == 'qflux':
                            print 'and q file:' + meanfile_q

                        if levsel:
                            ncout_u = mync.open_multi(meanfile_u, globv1, name2, \
                                                       dataset=dset2, subs=sub, levsel=levc)
                            ncout_v = mync.open_multi(meanfile_v, globv2, name2, \
                                                       dataset=dset2, subs=sub, levsel=levc)

                            if variable == 'qflux':
                                 ncout_q = mync.open_multi(meanfile_q, globv3, name2, \
                                                           dataset=dset2, subs=sub, levsel=levc)

                        else:
                            ncout_u = mync.open_multi(meanfile_u, globv1, name2, \
                                                       dataset=dset2, subs=sub)
                            ncout_v = mync.open_multi(meanfile_v, globv2, name2, \
                                                       dataset=dset2, subs=sub)

                            if variable == 'qflux':
                                ncout_q = mync.open_multi(meanfile_q, globv3, name2, \
                                                           dataset=dset2, subs=sub)


                        ndim = len(ncout_u)

                        if ndim == 5:

                            meandata_u, time, lat, lon, dtime = ncout_u
                            meandata_v, time, lat, lon, dtime = ncout_v

                            if variable == 'qflux':
                                meandata_q, time, lat, lon, meandtime = ncout_q

                        elif ndim == 6:

                             meandata_u, time, lat, lon, lev, dtime = ncout_u
                             meandata_u = np.squeeze(meandata_u)

                             meandata_v, time, lat, lon, lev, dtime = ncout_v
                             meandata_v = np.squeeze(meandata_v)

                             if variable == 'qflux':
                                 meandata_q, time, lat, lon, lev, dtime = ncout_q
                                 meandata_q = np.squeeze(meandata_q)

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
                            meandata_u = meandata_u[:, :, ord]
                            meandata_v = meandata_v[:, :, ord]

                            if variable == 'qflux':
                                meandata_q = meandata_q[:, :, ord]

                    # If anomaly for contour get the mean
                    if pluscon:
                        if ctyp_con == 'anom_mon' or ctyp_con=='anom_seas':
                            meanfile_c = bkdir + 'metbot_multi_dset/' + dset3 + '/' + name3 + '/' \
                                         + name3 + '.' + globv_c + '.mon.mean.' + ysclim_c + '.nc'

                            if levcon:
                                ncout_c = mync.open_multi(meanfile_c, globv_c, name3, \
                                                          dataset=dset3, subs=sub, levsel=lev_c)
                            else:
                                ncout_c = mync.open_multi(meanfile_c, globv_c, name3, \
                                                          dataset=dset3, subs=sub)
                            ndim_c = len(ncout_c)

                            if ndim_c == 5:

                                meandata_c, time_c, lat_c, lon_c, dtime_c = ncout_c

                            elif ndim_c == 6:

                                meandata_c, time_c, lat_c, lon_c, lev_c, dtime_c = ncout_c
                                meandata_c = np.squeeze(meandata_c)

                            dtime_c[:, 3] = 0

                            if domain == 'mac_wave' or domain == 'bigtrop':
                                print "Ammending lons around 0"
                                for i in range(len(lon_c)):
                                    if lon_c[i] > 180:
                                        lon_c[i] = lon_c[i] - 360
                                ord = np.argsort(lon_c)
                                lon_c = lon_c[ord]
                                meandata_c = meandata_c[:, :, ord]

                    # Interpolate data
                    if interp:
                        print "Interpolating data to a " + str(int_res) + " grid"
                        if name == 'cdr':  # hopefully this saves for all because it's the first model
                            minlon = lon[0]
                            maxlon = lon[nlon -1]
                            minlat = lat[0]
                            maxlat = lat[nlat - 1]

                        newlon = np.arange(minlon, maxlon + int_res, int_res)
                        newlat = np.arange(maxlat, minlat + int_res, int_res)
                        # newlat=newlat[::-1]

                        nlon = len(newlon)
                        nlat = len(newlat)

                        prodata_u = np.zeros((nsamp, nlat, nlon), dtype=np.float32)
                        prodata_v = np.zeros((nsamp, nlat, nlon), dtype=np.float32)

                        if variable == 'qflux':
                            prodata_q = np.zeros((nsamp, nlat, nlon), dtype=np.float32)

                        if pluscon:
                            prodata_c = np.zeros((nsamp, nlat, nlon), dtype=np.float32)

                        # Get rid of nans
                        nonan_u = np.nan_to_num(smpdata_u)
                        nonan_v = np.nan_to_num(smpdata_v)

                        if variable == 'qflux':
                            nonan_q = np.nan_to_num(smpdata_q)

                        if pluscon:
                            nonan_c = np.nan_to_num(smpdata_c)

                        for step in range(nsamp):
                            Interpolator_u = spi.interp2d(lon, lat, nonan_u[step, :, :], kind='linear')
                            Interpolator_v = spi.interp2d(lon, lat, nonan_v[step, :, :], kind='linear')

                            prodata_u[step, :, :] = Interpolator_u(newlon, newlat)
                            prodata_v[step, :, :] = Interpolator_v(newlon, newlat)

                            if variable == 'qflux':
                                Interpolator_q = spi.interp2d(lon, lat, nonan_q[step, :, :], kind='linear')
                                prodata_q[step, :, :] = Interpolator_q(newlon, newlat)

                        if pluscon:
                            for st in range(nsamp):
                                Interpolator_c = spi.interp2d(lon_c, lat_c, nonan_c[st, :, :], kind='linear')
                                prodata_c[st, :, :] = Interpolator_c(newlon, newlat)


                        # If manntest also interpolate the full file
                        if manntest:

                            proalldata_u = np.zeros((nsteps, nlat, nlon), dtype=np.float32)
                            proalldata_v = np.zeros((nsteps, nlat, nlon), dtype=np.float32)

                            if variable == 'qflux':
                                proalldata_q = np.zeros((nsteps, nlat, nlon), dtype=np.float32)

                            if pluscon:
                                proalldata_c = np.zeros((nsteps_c, nlat, nlon), dtype=np.float32)

                            # Get rid of nans
                            nonan_u = np.nan_to_num(alldata_u)
                            nonan_v = np.nan_to_num(alldata_v)

                            if variable == 'qflux':
                                nonan_q = np.nan_to_num(alldata_q)

                            if pluscon:
                                nonan_c = np.nan_to_num(alldata_c)

                            for step in range(nsteps):
                                Interpolator_u = spi.interp2d(lon, lat, nonan_u[step, :, :], kind='linear')
                                Interpolator_v = spi.interp2d(lon, lat, nonan_v[step, :, :], kind='linear')

                                proalldata_u[step, :, :] = Interpolator_u(newlon, newlat)
                                proalldata_v[step, :, :] = Interpolator_v(newlon, newlat)

                                if variable == 'qflux':
                                    Interpolator_q = spi.interp2d(lon, lat, nonan_q[step, :, :], kind='linear')
                                    proalldata_q[step, :, :] = Interpolator_q(newlon, newlat)

                            if pluscon:
                                for st in range(nsteps_c):
                                    Interpolator_c = spi.interp2d(lon_c, lat_c, nonan_c[st, :, :], kind='linear')
                                    proalldata_c[st, :, :] = Interpolator_c(newlon, newlat)

                        # If anomaly also interpolate the monthly means
                        if ctyp == 'anom_mon' or ctyp=='anom_seas':

                            promeandata_u = np.zeros((12, nlat, nlon), dtype=np.float32)
                            promeandata_v = np.zeros((12, nlat, nlon), dtype=np.float32)

                            if variable == 'qflux':
                                promeandata_q = np.zeros((12, nlat, nlon), dtype=np.float32)

                            # Get rid of nans
                            nonan_u = np.nan_to_num(meandata_u)
                            nonan_v = np.nan_to_num(meandata_v)

                            if variable == 'qflux':
                                nonan_q = np.nan_to_num(meandata_q)

                            for step in range(12):
                                Interpolator_u = spi.interp2d(lon, lat, nonan_u[step, :, :], kind='linear')
                                Interpolator_v = spi.interp2d(lon, lat, nonan_v[step, :, :], kind='linear')

                                promeandata_u[step, :, :] = Interpolator_u(newlon, newlat)
                                promeandata_v[step, :, :] = Interpolator_v(newlon, newlat)

                                if variable == 'qflux':
                                    Interpolator_q = spi.interp2d(lon, lat, nonan_q[step, :, :], kind='linear')
                                    promeandata_q[step, :, :] = Interpolator_q(newlon, newlat)



                        if pluscon:
                            if ctyp_con == 'anom_mon' or ctyp_con=='anom_seas':
                                promeandata_c = np.zeros((12, nlat, nlon), dtype=np.float32)
                                nonan_c = np.nan_to_num(meandata_c)

                                for st in range(12):
                                    Interpolator_c = spi.interp2d(lon_c, lat_c, nonan_c[st, :, :], kind='linear')
                                    promeandata_c[st, :, :] = Interpolator_c(newlon, newlat)

                    # if not interpolating
                    else:

                        prodata_u = smpdata_u
                        prodata_v = smpdata_v

                        if variable == 'qflux':
                            prodata_q = smpdata_q

                        if pluscon:
                            prodata_c = smpdata_c

                        newlon = lon
                        newlat = lat

                        if manntest:
                            proalldata_u=alldata_u
                            proalldata_v=alldata_v

                            if variable=='qflux':
                                proalldata_q=alldata_q

                            if pluscon:
                                proalldata_c=alldata_c

                        if ctyp=='anom_mon':
                            promeandata_u=meandata_u
                            promeandata_v=meandata_v

                            if variable=='qflux':
                                promeandata_q=meandata_q

                        if pluscon:
                            if ctyp_con == 'anom_mon':
                                promeandata_c=meandata_c

                    # If qflux then multiply sample winds with sample humidity
                    if variable == 'qflux':
                        print "Multiplying winds by q..."

                        qu = prodata_u * prodata_q
                        qv = prodata_v * prodata_q

                        prodata_u = qu
                        prodata_v = qv

                        if manntest:
                            qu_all = proalldata_u *proalldata_q
                            qv_all = proalldata_v *proalldata_q

                            proalldata_u=qu_all
                            proalldata_v=qv_all

                        if ctyp=='anom_mon' or ctyp=='anom_seas':

                            qu_mean= promeandata_u * promeandata_q
                            qv_mean= promeandata_v * promeandata_q

                            promeandata_u=qu_mean
                            promeandata_v=qv_mean


                    # Do Mann Whitney test
                    if manntest:

                        otherdata_u = proalldata_u[oinds]
                        otherdata_v = proalldata_v[oinds]

                        if variable == 'qflux':
                            otherdata_q = proalldata_q[oinds]

                            qu = otherdata_u * otherdata_q
                            qv = otherdata_v * otherdata_q

                            otherdata_u = qu
                            otherdata_v = qv

                        if pluscon:
                            otherdata_c = proalldata_c[oinds_c]

                        # Do mannwhitney test
                        uvals_u = np.zeros((nlat, nlon), dtype=np.float32)
                        pvals_u = np.zeros((nlat, nlon), dtype=np.float32)

                        uvals_v = np.zeros((nlat, nlon), dtype=np.float32)
                        pvals_v = np.zeros((nlat, nlon), dtype=np.float32)

                        for i in range(nlat):
                            for j in range(nlon):

                                smpbox_u = prodata_u[:, i, j]
                                otherbox_u = otherdata_u[:, i, j]
                                ustat, pvalue = scipy.stats.mannwhitneyu(smpbox_u, otherbox_u, alternative='two-sided')

                                uvals_u[i, j] = ustat
                                pvals_u[i, j] = pvalue

                                smpbox_v = prodata_v[:, i, j]
                                otherbox_v = otherdata_v[:, i, j]
                                ustat, pvalue = scipy.stats.mannwhitneyu(smpbox_v, otherbox_v, alternative='two-sided')

                                uvals_v[i, j] = ustat
                                pvals_v[i, j] = pvalue


                        if pluscon:
                            uvals_c = np.zeros((nlat, nlon), dtype=np.float32)
                            pvals_c = np.zeros((nlat, nlon), dtype=np.float32)

                            for i in range(nlat):
                                for j in range(nlon):

                                    smpbox_c = prodata_c[:, i, j]
                                    otherbox_c = otherdata_c[:, i, j]
                                    ustat, pvalue = scipy.stats.mannwhitneyu(smpbox_c, otherbox_c, alternative='two-sided')

                                    uvals_c[i, j] = ustat
                                    pvals_c[i, j] = pvalue

                        if not fdr:
                            # Simple approach - Get mask over values where pval <=0.1
                            mask_pvals = np.zeros((nlat, nlon), dtype=np.float32)
                            for i in range(nlat):
                                for j in range(nlon):
                                    if pvals_u[i, j] <= 0.1 or pvals_v[i,j] <= 0.1:
                                        mask_pvals[i, j] = 1
                                    else:
                                        mask_pvals[i, j] = 0

                            if pluscon:
                                mask_pvals_c=np.zeros((nlat, nlon), dtype=np.float32)
                                for i in range(nlat):
                                    for j in range(nlon):
                                        if pvals_c[i, j] <= 0.1:
                                            mask_pvals[i, j] = 1
                                        else:
                                            mask_pvals[i, j] = 0

                        else:

                            # Get p value that accounts for False Discovery Rate (FDR)
                            nboxes = nlat * nlon
                            gcnt = 1
                            plist_u = []
                            plist_v = []
                            for i in range(nlat):
                                for j in range(nlon):

                                    # first for u
                                    thisp = pvals_u[i, j]
                                    stat = (gcnt / float(nboxes)) * alphaFDR
                                    if thisp <= stat:
                                        plist_u.append(thisp)

                                    # then for v
                                    thisp = pvals_v[i, j]
                                    stat = (gcnt / float(nboxes)) * alphaFDR
                                    if thisp <= stat:
                                        plist_v.append(thisp)

                                    gcnt += 1

                            plist_u = np.asarray(plist_u)
                            plist_v = np.asarray(plist_v)
                            print plist_u
                            print plist_v
                            print len(plist_u)
                            print len(plist_v)
                            pmax_u = np.max(plist_u)
                            pmax_v = np.max(plist_v)
                            print 'pFDR = '
                            print pmax_u
                            print pmax_v

                            mask_pvals = np.zeros((nlat, nlon), dtype=np.float32)
                            for i in range(nlat):
                                for j in range(nlon):
                                    if pvals_u[i, j] <= pmax_u or pvals_v[i,j] <= pmax_v:
                                        mask_pvals[i, j] = 1
                                    else:
                                        mask_pvals[i, j] = 0

                            if pluscon:
                                gcnt = 1
                                plist_c=[]
                                for i in range(nlat):
                                    for j in range(nlon):

                                        thisp = pvals_c[i, j]
                                        stat = (gcnt / float(nboxes)) * alphaFDR
                                        if thisp <= stat:
                                            plist_c.append(thisp)
                                        gcnt += 1

                                plist_c = np.asarray(plist_c)
                                pmax_c = np.max(plist_c)

                                mask_pvals_c = np.zeros((nlat, nlon), dtype=np.float32)
                                for i in range(nlat):
                                    for j in range(nlon):
                                        if pvals_c[i, j] <= pmax_c:
                                            mask_pvals_c[i, j] = 1
                                        else:
                                            mask_pvals_c[i, j] = 0

                    if cnt==1:
                        m, f = pt.AfrBasemap(newlat, newlon, drawstuff=True, prj='cyl', fno=1, rsltn='l')


                    # Get lon lat grid
                    plon, plat = np.meshgrid(newlon, newlat)

                    # Get composite
                    print "Calculating composite..."
                    comp_u = np.nanmean(prodata_u, 0)
                    comp_v = np.nanmean(prodata_v, 0)

                    if pluscon:
                        comp_c = np.nanmean(prodata_c, 0)

                    compdata_u = np.squeeze(comp_u)
                    compdata_v = np.squeeze(comp_v)

                    if pluscon:
                        compdata_c = np.squeeze(comp_c)

                    if ctyp=='abs':

                        data4plot_u = compdata_u
                        data4plot_v = compdata_v

                        if manntest:
                            data4plot_u = data4plot_u * mask_pvals
                            data4plot_v = data4plot_v * mask_pvals


                    elif ctyp=='anom_mon':
                        print "Calculating anomaly from long term MONTHLY means..."

                        anoms_u=np.zeros((int(nsamp),nlat,nlon),dtype=np.float32)
                        anoms_v = np.zeros((int(nsamp), nlat, nlon), dtype=np.float32)
                        for day in range(nsamp):
                            mon_thisday=smpdtime[day,1]

                            this_monmean_u=promeandata_u[mon_thisday-1]
                            this_anom_u=prodata_u[day,:,:]-this_monmean_u
                            anoms_u[day,:,:]=this_anom_u

                            this_monmean_v=promeandata_v[mon_thisday-1]
                            this_anom_v=prodata_v[day,:,:]-this_monmean_v
                            anoms_v[day,:,:]=this_anom_v

                        anom_comp_u=np.nanmean(anoms_u,0)
                        anom_comp_v=np.nanmean(anoms_v,0)

                        data4plot_u = anom_comp_u
                        data4plot_v = anom_comp_v

                    elif ctyp == 'anom_seas':

                        # get seasonal mean
                        thesemons_u = np.zeros((nmon, nlat, nlon), dtype=np.float32)
                        thesemons_v = np.zeros((nmon, nlat, nlon), dtype=np.float32)
                        for zz in range(len(mons)):
                            thesemons_u[zz, :, :] = promeandata_u[mons[zz] - 1, :, :]
                            thesemons_v[zz, :, :] = promeandata_v[mons[zz] - 1, :, :]
                        seasmean_u = np.nanmean(thesemons_u, 0)
                        seasmean_v = np.nanmean(thesemons_v, 0)

                        anoms_u = np.asarray([prodata_u[x, :, :] - seasmean_u for x in range(len(prodata_u[:, 0, 0]))])
                        anoms_v = np.asarray([prodata_v[x, :, :] - seasmean_v for x in range(len(prodata_v[:, 0, 0]))])

                        anom_comp_u = np.nanmean(anoms_u, 0)
                        anom_comp_v = np.nanmean(anoms_v, 0)

                        data4plot_u = anom_comp_u
                        data4plot_v = anom_comp_v

                    if ctyp=='anom_mon' or ctyp=='anom_seas':
                        if agtest:

                            print "Calculating number of days which agree..."

                            # Get signs
                            anoms_signs_u = np.sign(anoms_u)
                            anoms_signs_v = np.sign(anoms_v)

                            comp_signs_u = np.sign(anom_comp_u)
                            comp_signs_v = np.sign(anom_comp_v)

                            # Count ndays in composite with same sign as mean
                            mask_zeros = np.zeros((nlat, nlon), dtype=np.float32)
                            for i in range(nlat):
                                for j in range(nlon):
                                    count_u = len(np.where(anoms_signs_u[:, i, j] == comp_signs_u[i, j])[0])
                                    count_v = len(np.where(anoms_signs_v[:, i, j] == comp_signs_v[i, j])[0])

                                    perc_u = (float(count_u) / float(nsamp)) * 100
                                    perc_v = (float(count_v) / float(nsamp)) * 100

                                    if perc_u >= perc_ag or perc_v >= perc_ag:
                                        mask_zeros[i, j] = 1
                                    else:
                                        mask_zeros[i, j] = 0

                            # Set masked as 0
                            zeroed_comp_u = anom_comp_u * mask_zeros
                            zeroed_comp_v = anom_comp_v * mask_zeros

                            # Set masked as mask!
                            # masked_comp=np.ma.masked_where(mask_zeros<1,anom_comp)

                            data4plot_u = zeroed_comp_u
                            data4plot_v = zeroed_comp_v

                        elif manntest:

                            mwzero_comp_u = anom_comp_u * mask_pvals
                            mwzero_comp_v = anom_comp_v * mask_pvals

                            data4plot_u=mwzero_comp_u
                            data4plot_v=mwzero_comp_v

                    if pluscon:
                        if ctyp_con=='abs':
                            data4plot_c = compdata_c

                            if manntest:
                                data4plot_c=data4plot_c*mask_pvals_c

                        elif ctyp_con=='anom_mon':

                            anoms_c = np.zeros((int(nsamp), nlat, nlon), dtype=np.float32)
                            for day in range(nsamp):

                                mon_thisday = smpdtime_c[day, 1]

                                this_monmean_c = promeandata_c[mon_thisday - 1]
                                this_anom_c = prodata_c[day, :, :] - this_monmean_c
                                anoms_c[day, :, :] = this_anom_c

                            anom_comp_c = np.nanmean(anoms_c, 0)

                            data4plot_c = anom_comp_c

                        elif ctyp_con=='anom_seas':

                            # get seasonal mean
                            thesemons = np.zeros((nmon, nlat, nlon), dtype=np.float32)
                            for zz in range(len(mons)):
                                thesemons[zz, :, :] = promeandata_c[mons[zz] - 1, :, :]
                            seasmean = np.nanmean(thesemons, 0)

                            anoms_c = np.asarray([prodata_c[x, :, :] - seasmean for x in range(len(prodata_c[:, 0, 0]))])

                            anom_comp_c = np.nanmean(anoms_c, 0)
                            data4plot_c = anom_comp_c


                        if ctyp_con =='anom_mon' or ctyp_con=='anom_seas':

                            if agtest_con:
                                anoms_signs = np.sign(anoms_c)
                                comp_signs = np.sign(anom_comp_c)

                                mask_zeros = np.zeros((nlat, nlon), dtype=np.float32)
                                for i in range(nlat):
                                    for j in range(nlon):
                                        count = len(np.where(anoms_signs[:, i, j] == comp_signs[i, j])[0])
                                        perc = (float(count) / float(nsamp)) * 100
                                        if perc >= perc_ag_con:
                                            mask_zeros[i, j] = 1
                                        else:
                                            mask_zeros[i, j] = 0

                                zeroed_comp_c = anom_comp_c * mask_zeros

                                data4plot_c= zeroed_comp_c

                            if manntest:

                                data4plot_c = anom_comp_c*mask_pvals_c


                    # Add skips
                    data4plot_u=data4plot_u[::skip,::skip]
                    data4plot_v = data4plot_v[::skip, ::skip]
                    newlon=newlon[::skip]
                    newlat=newlat[::skip]

                    # Plot
                    print "Plotting for model "+name2
                    plt.subplot(yplots,xplots,cnt)

                    # Plot contours if pluscon
                    if pluscon:
                        if globv_c == 'olr':
                            clevs = np.arange(200, 280, 10)
                            cm = plt.cm.Wistia_r
                        elif globv_c == 'omega':
                            clevs = np.arange(-0.12, 0.14, 0.02)
                            cm = plt.cm.bwr
                        elif globv_c=='q':
                            clevs = np.arange(-0.002, 0.00225, 0.00025)
                            #cm = plt.cm.bwr_r
                            cm = plt.cm.BrBG
                        else:
                            print "Need to specify cbar for this variable"

                        cs = m.contourf(plon, plat, data4plot_c, clevs, cmap=cm, extend='both')


                    # Plot vectors if not
                    if variable == 'wind':
                        if choosel[l] == '850':
                            wind_sc = 175
                            usc = 10
                            lab = '10m/s'
                        elif choosel[l] == '200':
                            if ctyp=='abs':
                                if domain=='swio':
                                    wind_sc = 600
                                    usc = 40
                                    lab = '40m/s'
                                elif domain=='mac_wave':
                                    wind_sc = 600
                                    usc = 40
                                    lab = '40m/s'
                                elif domain=='nglob':
                                    wind_sc = 850
                                    usc = 50
                                    lab = '50m/s'
                            elif ctyp=='anom_mon' or ctyp=='anom_seas':
                                if domain == 'swio':
                                    # wind_sc = 180
                                    # usc = 10
                                    # lab = '10m/s'
                                    wind_sc=180
                                    usc= 10
                                    lab= '10m/s'

                                elif domain =='mac_wave':
                                    wind_sc = 180
                                    usc = 10
                                    lab = '10m/s'
                                elif domain=='nglob':
                                    wind_sc = 350
                                    usc = 30
                                    lab = '30m/s'
                    elif variable == 'qflux':
                        if choosel[l] == '850':
                            if ctyp == 'abs':
                                wind_sc = 0.6
                                usc = 0.075
                                lab = '0.075 kg/kg/ms'
                            elif ctyp == 'anom_mon' or ctyp=='anom_seas':
                                wind_sc = 0.4
                                usc = 0.05
                                lab = '0.05 kg/kg/ms'

                    if ctyp=='anom_mon' or ctyp=='anom_seas':
                        q = plt.quiver(newlon, newlat, data4plot_u, data4plot_v, scale=wind_sc, width=0.005)
                    elif ctyp=='abs':
                        q = plt.quiver(newlon, newlat, data4plot_u, data4plot_v, scale=wind_sc)
                    if cnt==1:
                        plt.quiverkey(q, X=1.2, Y=1.1, U=usc, label=lab, labelpos='W', fontproperties={'size': 'xx-small'})

                    if dset == 'noaa':
                        pltname = name + '/' + name2
                    else:
                        pltname = name
                    plt.title(pltname, fontsize=8, fontweight='demibold')

                    # Redraw map
                    m.drawcountries()
                    m.drawcoastlines()

                    cnt += 1


            print "Finalising plot..."
            plt.subplots_adjust(left=0.05,right=0.9,top=0.95,bottom=0.02,wspace=0.1,hspace=0.2)

            if pluscon:
                # Plot cbar
                axcl = g.add_axes([0.91, 0.15, 0.01, 0.6])
                cbar = plt.colorbar(cs, cax=axcl)
                my.ytickfonts(fontsize=12.)



            if pluscon:
                vfname=variable+'_'+globv_c
            else:
                vfname=variable

            # Save
            if ctyp=='anom_mon' or ctyp=='anom_seas':
                if agtest:
                    cstr=ctyp+'_agtest_'+str(perc_ag)
                else:
                    cstr=ctyp
            else:
                cstr=ctyp

            if manntest:
                cstr = cstr + 'manntest_' + str(alphaFDR)

            if climyr=='spec':
                cstr = cstr + '_35years_'


            if lag:
                compname = figdir + 'multi_comp_'+cstr+'.'+sample+'.' + type + '.' + vfname + \
                          '.'+choosel[l]+'.'+sub+'.from_event'+from_event+'.'+str(int_res)+'.lag_'+str(edays[lo])+'.'+rean+'.skip'+str(skip)+'.png'
            else:
                compname = figdir + 'multi_comp_'+cstr+'.'+sample+'.' + type + '.' + vfname + \
                      '.'+choosel[l]+'.'+sub+'.from_event'+from_event+'.'+str(int_res)+'.'+rean+'.skip'+str(skip)+'.png'

            plt.savefig(compname, dpi=150)
            plt.close()