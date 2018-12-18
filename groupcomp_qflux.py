# To plot composites of mean qflux for each group
# This one is for qflux and q at 850
# but now reading in netcdf files to speed things up
#
#
# edit


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
import MetBot.dset_dict as dsetdict
import MetBot.mast_dset_dict as mast_dict
import MetBot.dimensions_dict as dim_exdict
import dsets_mplot_5group_4plot_qflux as dset_mp
import MetBot.mytools as my
import MetBot.mynetcdf as mync
import PlotTools as pt
import scipy


### Running options
test_scr=False
xplots = 3
yplots = 2
spec_col=True
sub='SA_qflux'
#agtest=False # do a test on the proportion of days in comp that agree in dir change
#perc_ag=75 # show if this % or more days agree

rean='era' # reanalysis - ncep or era
interp=True
int_res=1.75

seas='DJF'
climyr='spec' # this is to use new climatology files which are based on only 35 years
                # 'spec' is 35 years
                # 'prev' is previous files - all different climatologies
# Info for vector
varlist=['qflux']
ctyp='anom' #abs is absolute,  anom is with reference data deducted for the models
levsel=True
if levsel:
    choosel=['850'] # can add a list
else:
    choosel=['1']


if ctyp=='abs':
    skip=1
elif ctyp=='anom':
    skip=2 # to skip over some vectors in plotting. If want to skip none use 1

# Info for contour
pluscon=True
convar=['q']
ctyp_con='anom'
levcon=True
if levcon:
    chooselc=['850'] # can add a list
else:
    chooselc=['1']
# agtest_con=False
# perc_ag_con=75

figdim = [11,5]

### Get directories
bkdir=cwd+"/../../CTdata/"
thisdir=bkdir+"/hpaperplot/"
botdir=bkdir+"metbot_multi_dset/"

figdir=thisdir+"groupcomp_qflux/"
my.mkdir_p(figdir)

if seas == 'DJF':
    mons=[1,2,12]
    mon1 = 12
    mon2 = 2
    nmon = 3

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

groups = ['fuchsia', 'gold', 'darkblue', 'r', 'blueviolet', 'springgreen']
ngroup = [1,7,8,3,6,2]
grname = ['Reference','Yellow','Navy','Red','Purple','Green']

# Set up plot
print "Setting up plot..."
w, ax = plt.subplots(figsize=figdim)

cnt = 1

# First we do everything for CDR
### Dsets
dsets = 'all'
dsetnames = ['noaa']
ndset = len(dsetnames)
ndstr = str(ndset)

print "Looping datasets"
for d in range(ndset):
    dset = dsetnames[d]
    dcnt = str(d + 1)

    if dset != 'cmip5':
        levc = int(choosel[l])
    else:
        levc = int(choosel[l]) * 100

    if pluscon:
        if dset != 'cmip5':
            lev_c = int(chooselc[l])
        else:
            lev_c = int(chooselc[l]) * 100

    ### Models
    mods = 'all'
    nmod = len(dset_mp.dset_deets[dset])
    mnames = list(dset_mp.dset_deets[dset])
    nmstr = str(nmod)

    for mo in range(nmod):
        name = mnames[mo]
        groupdct = dset_mp.dset_deets[dset][name]
        thisgroup = int(groupdct['group'])
        if thisgroup == 1:

            mcnt = str(mo + 1)
            print 'Running on ' + name
            print 'This is model ' + mcnt + ' of ' + nmstr + ' in list'

            # Switch variable if NOAA
            if dset == 'noaa':
                if rean == 'ncep':
                    ds4noaa = 'ncep'
                    mod4noaa = 'ncep2'
                elif rean == 'era':
                    ds4noaa = 'era'
                    mod4noaa = 'erai'
                dset2 = ds4noaa
                name2 = mod4noaa
            else:
                dset2 = dset
                name2 = name

            if pluscon:
                if dset == 'noaa':
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
                conmastdct = mast_dict.mast_dset_deets[dset3]

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

            # Years for clim and manntest
            if climyr == 'spec':
                ysclim = moddct['yrfname']
                year1 = float(ysclim[0:4])
                year2 = float(ysclim[5:9])

            if pluscon:
                if climyr == 'spec':
                    ysclim_c = condct['yrfname']
                    year1_c = float(ysclim_c[0:4])
                    year2_c = float(ysclim_c[5:9])

            dimdict = dim_exdict.dim_deets[globv1][dset2]
            latname = dimdict[1]
            lonname = dimdict[2]

            if pluscon:
                dimdict_c = dim_exdict.dim_deets[globv_c][dset3]
                latname_c = dimdict_c[1]
                lonname_c = dimdict_c[2]

            meanfile_u = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
                         + name2 + '.' + globv1 + '.mon.mean.' + ysclim + '.nc'

            meanfile_v = bkdir + 'metbot_multi_dset/' + dset2 + '/' + name2 + '/' \
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

            nlon=len(lon)
            nlat=len(lat)

            # If anomaly for contour get the mean
            if pluscon:
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
                    promeandata_c = np.zeros((12, nlat, nlon), dtype=np.float32)
                    nonan_c = np.nan_to_num(meandata_c)

                    for st in range(12):
                        Interpolator_c = spi.interp2d(lon_c, lat_c, nonan_c[st, :, :], kind='linear')
                        promeandata_c[st, :, :] = Interpolator_c(newlon, newlat)

            # if not interpolating
            else:

                newlon = lon
                newlat = lat

                promeandata_u = meandata_u
                promeandata_v = meandata_v

                if variable == 'qflux':
                    promeandata_q = meandata_q

                if pluscon:
                    promeandata_c = meandata_c

            # If qflux then multiply sample winds with sample humidity
            if variable == 'qflux':
                print "Multiplying winds by q..."

                qu_mean = promeandata_u * promeandata_q
                qv_mean = promeandata_v * promeandata_q

                promeandata_u = qu_mean
                promeandata_v = qv_mean

            # get seasonal mean
            thesemons_u = np.zeros((nmon, nlat, nlon), dtype=np.float32)
            thesemons_v = np.zeros((nmon, nlat, nlon), dtype=np.float32)
            for zz in range(len(mons)):
                thesemons_u[zz, :, :] = promeandata_u[mons[zz] - 1, :, :]
                thesemons_v[zz, :, :] = promeandata_v[mons[zz] - 1, :, :]
            seasmean_u = np.nanmean(thesemons_u, 0)
            seasmean_v = np.nanmean(thesemons_v, 0)

            if pluscon:
                # get seasonal mean
                thesemons = np.zeros((nmon, nlat, nlon), dtype=np.float32)
                for zz in range(len(mons)):
                    thesemons[zz, :, :] = promeandata_c[mons[zz] - 1, :, :]
                seasmean_c = np.nanmean(thesemons, 0)

            if name == 'cdr':
                ref_u = seasmean_u
                ref_v = seasmean_v
                ref_c = seasmean_c

            if cnt == 1:
                m, f = pt.AfrBasemap(newlat, newlon, drawstuff=True, prj='cyl', fno=1, rsltn='l')

            # Get lon lat grid
            plon, plat = np.meshgrid(newlon, newlat)

            data4plot_u = seasmean_u
            data4plot_v = seasmean_v

            if pluscon:
                data4plot_c = seasmean_c


            # Add skips
            data4plot_u = data4plot_u[::skip, ::skip]
            data4plot_v = data4plot_v[::skip, ::skip]
            newlon = newlon[::skip]
            newlat = newlat[::skip]

            # Plot
            print "Plotting for model " + name2
            plt.subplot(yplots, xplots, cnt)

            # Plot contours if pluscon
            if pluscon:
                if globv_c == 'q':
                    clevs = np.arange(0.0, 0.02, 0.002)
                    # cm = plt.cm.bwr_r
                    #cm = plt.cm.YlGnBu
                    #cm = plt.cm.BuPu
                    cm = plt.cm.BrBG
                else:
                    print "Need to specify cbar for this variable"

                cs = m.contourf(plon, plat, data4plot_c, clevs, cmap=cm, extend='both')

            # Plot vectors
            if variable == 'qflux':
                if choosel[l] == '850':
                    wind_sc = 0.6
                    usc = 0.075
                    lab = '0.075 kg/kg/ms'

            q = plt.quiver(newlon, newlat, data4plot_u, data4plot_v, scale=wind_sc, width=0.005)
            if cnt == 1:
                plt.quiverkey(q, X=0.9, Y=1.1, U=usc, label=lab, labelpos='W', fontproperties={'size': 'xx-small'})

            pltname= name2
            plt.title(pltname, fontsize=8, fontweight='demibold')

            # Redraw map
            m.drawcountries()
            m.drawcoastlines()

            m.drawmapboundary(color=groups[thisgroup-1], linewidth=3)


plt.subplots_adjust(left=0.05,right=0.8,top=0.95,bottom=0.02,wspace=0.1,hspace=0.2)
if pluscon:
    # Plot cbar
    axcl = w.add_axes([0.81, 0.15, 0.01, 0.6])
    cbar = plt.colorbar(cs, cax=axcl)
    my.ytickfonts(fontsize=10.,fontweight='demibold')

cnt += 1

# Loop groups
grcnt = np.zeros(5, dtype=np.int8)
for g in range(len(groups)-1):

    gind=g+1 # to skip CDR
    gnum=g+2

    grcollect_u=np.zeros((ngroup[gind], nlat, nlon), dtype=np.float32)
    grcollect_v=np.zeros((ngroup[gind], nlat, nlon), dtype=np.float32)

    if pluscon:
        grcollect_c=np.zeros((ngroup[gind], nlat, nlon), dtype=np.float32)


    ### Dsets
    dsets='all'
    dsetnames=['cmip5']
    ndset=len(dsetnames)
    ndstr=str(ndset)

    print "Looping datasets"
    for d in range(ndset):
        dset=dsetnames[d]
        dcnt=str(d+1)

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
        mnames = list(dset_mp.dset_deets[dset])
        nmstr = str(nmod)

        for mo in range(nmod):
            name = mnames[mo]
            groupdct = dset_mp.dset_deets[dset][name]
            thisgroup = int(groupdct['group'])
            if thisgroup==gnum:

                mcnt = str(mo + 1)
                print 'Running on ' + name
                print 'This is model ' + mcnt + ' of ' + nmstr + ' in list'

                dset2 = dset
                name2 = name

                if pluscon:
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


                # Years for clim and manntest
                if climyr == 'spec':
                    ysclim = moddct['yrfname']
                    year1 = float(ysclim[0:4])
                    year2 = float(ysclim[5:9])

                if pluscon:
                    if climyr == 'spec':
                        ysclim_c = condct['yrfname']
                        year1_c = float(ysclim_c[0:4])
                        year2_c = float(ysclim_c[5:9])

                dimdict = dim_exdict.dim_deets[globv1][dset2]
                latname = dimdict[1]
                lonname = dimdict[2]

                if pluscon:
                    dimdict_c = dim_exdict.dim_deets[globv_c][dset3]
                    latname_c = dimdict_c[1]
                    lonname_c = dimdict_c[2]

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


                # If anomaly for contour get the mean
                if pluscon:
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
                        promeandata_c = np.zeros((12, nlat, nlon), dtype=np.float32)
                        nonan_c = np.nan_to_num(meandata_c)

                        for st in range(12):
                            Interpolator_c = spi.interp2d(lon_c, lat_c, nonan_c[st, :, :], kind='linear')
                            promeandata_c[st, :, :] = Interpolator_c(newlon, newlat)

                # if not interpolating
                else:

                    newlon = lon
                    newlat = lat

                    promeandata_u=meandata_u
                    promeandata_v=meandata_v

                    if variable=='qflux':
                        promeandata_q=meandata_q

                    if pluscon:
                        promeandata_c=meandata_c

                # If qflux then multiply sample winds with sample humidity
                if variable == 'qflux':
                    print "Multiplying winds by q..."

                    qu_mean= promeandata_u * promeandata_q
                    qv_mean= promeandata_v * promeandata_q

                    promeandata_u=qu_mean
                    promeandata_v=qv_mean

                # get seasonal mean
                thesemons_u = np.zeros((nmon, nlat, nlon), dtype=np.float32)
                thesemons_v = np.zeros((nmon, nlat, nlon), dtype=np.float32)
                for zz in range(len(mons)):
                    thesemons_u[zz, :, :] = promeandata_u[mons[zz] - 1, :, :]
                    thesemons_v[zz, :, :] = promeandata_v[mons[zz] - 1, :, :]
                seasmean_u = np.nanmean(thesemons_u, 0)
                seasmean_v = np.nanmean(thesemons_v, 0)

                if pluscon:
                    # get seasonal mean
                    thesemons = np.zeros((nmon, nlat, nlon), dtype=np.float32)
                    for zz in range(len(mons)):
                        thesemons[zz, :, :] = promeandata_c[mons[zz] - 1, :, :]
                    seasmean_c = np.nanmean(thesemons, 0)

                grcollect_u[grcnt[g],:,:]=seasmean_u
                grcollect_v[grcnt[g],:,:]=seasmean_v

                if pluscon:
                    grcollect_c[grcnt[g],:,:]=seasmean_c

                grcnt[g]+=1

    data4plot_u=np.nanmean(grcollect_u,0)
    data4plot_v=np.nanmean(grcollect_v,0)
    if pluscon:
        data4plot_c=np.nanmean(grcollect_c,0)

    if ctyp=='anom':
        data4plot_u=data4plot_u-ref_u
        data4plot_v=data4plot_v-ref_v

    if pluscon:
        if ctyp_con=='anom':
            data4plot_c=data4plot_c-ref_c

    # Get lon lat grid
    plon, plat = np.meshgrid(newlon, newlat)

    # Add skips
    data4plot_u = data4plot_u[::skip, ::skip]
    data4plot_v = data4plot_v[::skip, ::skip]
    newlon = newlon[::skip]
    newlat = newlat[::skip]

    # Plot
    print "Plotting for model " + name2
    plt.subplot(yplots, xplots, cnt)

    # Plot contours if pluscon
    if pluscon:
        if globv_c == 'q':
            clevs = np.arange(-0.004, 0.0044, 0.0004)
            # cm = plt.cm.bwr_r
            cm = plt.cm.BrBG
        else:
            print "Need to specify cbar for this variable"

        cs = m.contourf(plon, plat, data4plot_c, clevs, cmap=cm, extend='both')

    # Plot vectors
    if variable == 'qflux':
        if choosel[l] == '850':
            if ctyp == 'abs':
                wind_sc = 0.6
                usc = 0.075
                lab = '0.075 kg/kg/ms'
            elif ctyp == 'anom':
                wind_sc = 0.4
                usc = 0.05
                lab = '0.05 kg/kg/ms'

    if ctyp == 'anom':
        q = plt.quiver(newlon, newlat, data4plot_u, data4plot_v, scale=wind_sc, width=0.005)
    elif ctyp == 'abs':
        q = plt.quiver(newlon, newlat, data4plot_u, data4plot_v, scale=wind_sc)
    if cnt == 2:
        plt.quiverkey(q, X=0.9, Y=1.1, U=usc, label=lab, labelpos='W', fontproperties={'size': 'xx-small'})

    pltname = grname[gind]
    plt.title(pltname, fontsize=8, fontweight='demibold')

    # Redraw map
    m.drawcountries()
    m.drawcoastlines()

    m.drawmapboundary(color=groups[gind], linewidth=3)

    cnt+=1

print "Finalising plot..."
plt.subplots_adjust(left=0.05,right=0.8,top=0.95,bottom=0.02,wspace=0.1,hspace=0.2)

if pluscon:
    # Plot cbar
    axcl = w.add_axes([0.91, 0.15, 0.01, 0.6])
    cbar = plt.colorbar(cs, cax=axcl)
    my.ytickfonts(fontsize=10.,fontweight='demibold')

if pluscon:
    vfname=variable+'_'+globv_c
else:
    vfname=variable

# Save
cstr=ctyp

if climyr=='spec':
    cstr = cstr + '_35years_'

compname = figdir + 'multi_comp_'+cstr+'.' + vfname + \
      '.'+choosel[l]+'.'+sub+'.'+str(int_res)+'.'+rean+'.skip'+str(skip)+'.png'

plt.savefig(compname, dpi=150)
plt.close()
