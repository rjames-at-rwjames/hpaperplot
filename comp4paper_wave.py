# To plot all CMIP5 models in multi-panel plot
# This one is for wind at 200 and omega 500
# so only 24 mods because of missing data
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
import dsets_paper_omega as dset_mp
import MetBot.dset_dict as dsetdict
import MetBot.mast_dset_dict as mast_dict
import MetBot.dimensions_dict as dim_exdict
import MetBot.mytools as my
import MetBot.mynetcdf as mync
import PlotTools as pt

### Running options
xplots = 4
yplots = 6
runs=['opt2']
wcb=['cont'] # which cloud band composite? Options: cont, mada, dbl
spec_col=True
domain='mac_wave' # swio or mac_wave
agtest=True # do a test on the proportion of days in comp that agree in dir change
perc_ag=75 # show if this % or more days agree
lag=True
thname='actual'
rean='ncep' # reanalysis - ncep or era
alphord=True
interp=True
int_res=3.5

# Info for vector
varlist=['wind']
ctyp='anom_mon' # abs or anom_mon
levsel=True
if levsel:
    choosel=['200'] # can add a list
else:
    choosel=['1']

# Info for contour
if ctyp=='abs':
    skip=2
elif ctyp=='anom_mon':
    skip=1 # to skip over some vectors in plotting. If want to skip none use 1
pluscon=True
convar=['omega']
ctyp_con='anom_mon'
levcon=True
if levcon:
    chooselc=['500'] # can add a list
else:
    chooselc=['1']
agtest_con=True
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
    figdim=[9,9]
elif domain=='nglob':
    sub='bigtrop'
elif domain=='mac_wave':
    sub='SASA'
    figdim = [9, 7]

### Get directories
bkdir=cwd+"/../../CTdata/"
thisdir=bkdir+"/hpaperplot/"
botdir=bkdir+"metbot_multi_dset/"

figdir=thisdir+"comp4paper_wave/"
if lag:
    figdir=thisdir+"comp4paper_wave_lags/"
my.mkdir_p(figdir)

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
                            ys_c = moddct['yrfname']
                        else:
                            if name3 == "MIROC5":
                                if globv_c == 'q':
                                    ys_c = moddct['fullrun']
                                elif globv_c == 'omega' or globv_c == 'gpth':
                                    ys_c = '1950_2009'
                                else:
                                    print 'variable ' + globv_c + ' has unclear yearname for ' + name3
                            else:
                                ys_c = moddct['fullrun']

                        print "Need to check which yfname using for contour..."
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
                        smpfile_q = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 +'/' \
                                +name+'.'+name2+'.' + globv3 + '.sampled_days.'\
                                + sample + '.'+from_event+'.'+type+'.'+thname+'.nc'
                    if pluscon:
                        smpfile_c = bkdir + 'metbot_multi_dset/' + dset3 + '/' + name3 +'/' \
                                +name+'.'+name3+'.' + globv_c + '.sampled_days.'\
                                + sample + '.'+from_event+'.'+type+'.'+thname+'.nc'


                    print 'Opening ' + smpfile_u
                    print 'and corresponding file: ' + smpfile_v
                    if variable == 'qflux':
                        print 'and q file:' + smpfile_q
                    if pluscon:
                        print 'and file for '+globv_c +' ' +smpfile_c

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

                    # If anom open longterm mean files
                    if ctyp == 'anom_mon':

                        meanfile_u=bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
                                       + name2 + '.' + globv1 + '.mon.mean.' + ys + '.nc'

                        meanfile_v=bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
                                       + name2 + '.' + globv2 + '.mon.mean.' + ys + '.nc'

                        if variable == 'qflux':
                            meanfile_q = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
                                         + name2 + '.' + globv3 + '.mon.mean.' + ys + '.nc'



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
                        if ctyp_con == 'anom_mon':
                            meanfile_c = bkdir + 'metbot_multi_dset/' + dset3 + '/' + name3 + '/' \
                                         + name3 + '.' + globv_c + '.mon.mean.' + ys_c + '.nc'

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
                            maxlon = lon[nlon - 1]
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

                        # If anomaly also interpolate the monthly means
                        if ctyp == 'anom_mon':

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
                            if ctyp_con == 'anom_mon':
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

                        if ctyp=='anom_mon':

                            qu_mean= promeandata_u * promeandata_q
                            qv_mean= promeandata_v * promeandata_q

                            promeandata_u=qu_mean
                            promeandata_v=qv_mean

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

                    if ctyp=='anom_mon':
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

                    if pluscon:
                        if ctyp_con=='abs':
                            data4plot_c = compdata_c
                        elif ctyp_con=='anom_mon':

                            anoms_c = np.zeros((int(nsamp), nlat, nlon), dtype=np.float32)
                            for day in range(nsamp):

                                mon_thisday = smpdtime_c[day, 1]

                                this_monmean_c = promeandata_c[mon_thisday - 1]
                                this_anom_c = prodata_c[day, :, :] - this_monmean_c
                                anoms_c[day, :, :] = this_anom_c

                            anom_comp_c = np.nanmean(anoms_c, 0)

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

                            else:

                                data4plot_c = anom_comp_c

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
                        else:
                            clevs = np.arange(-0.12, 0.14, 0.02)
                            cm = plt.cm.bwr
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
                            elif ctyp=='anom_mon':
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
                                wind_sc = 0.75
                                usc = 0.1
                                lab = '0.1 kg/kg/ms'
                            elif ctyp == 'anom_mon':
                                wind_sc = 0.4
                                usc = 0.05
                                lab = '0.05 kg/kg/ms'

                    if ctyp=='anom_mon':
                        q = plt.quiver(newlon, newlat, data4plot_u, data4plot_v, scale=wind_sc, width=0.005)
                    elif ctyp=='abs':
                        q = plt.quiver(newlon, newlat, data4plot_u, data4plot_v, scale=wind_sc)
                    if cnt==1:
                        plt.quiverkey(q, X=0.9, Y=1.1, U=usc, label=lab, labelpos='W', fontproperties={'size': 'xx-small'})

                    plt.title(name2, fontsize=8,fontweight='demibold')

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
            if ctyp=='anom_mon':
                if agtest:
                    cstr=ctyp+'_agtest_'+str(perc_ag)
                else:
                    cstr=ctyp
            else:
                cstr=ctyp
            if lag:
                compname = figdir + 'multi_comp_'+cstr+'.'+sample+'.' + type + '.' + vfname + \
                          '.'+choosel[l]+'.'+sub+'.from_event'+from_event+'.'+str(int_res)+'.lag_'+str(edays[lo])+'.'+rean+'.png'
            else:
                compname = figdir + 'multi_comp_'+cstr+'.'+sample+'.' + type + '.' + vfname + \
                      '.'+choosel[l]+'.'+sub+'.from_event'+from_event+'.'+str(int_res)+'.'+rean+'.png'
            plt.savefig(compname, dpi=150)
            plt.close()
