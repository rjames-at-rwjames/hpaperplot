# Bar chart plotting wrapper
#   to plot a bar chart
#   with colours for each group
#   first nttt


import numpy as np
import matplotlib.pyplot as plt
import sys,os
cwd=os.getcwd()
sys.path.append(cwd+'/..')
sys.path.append(cwd+'/../MetBot')
sys.path.append(cwd+'/../RTools')
sys.path.append(cwd+'/../quicks')
import MetBot.SynopticAnatomy as sy
import MetBot.EventStats as stats
import MetBot.SynopticPlot as syp
import MetBot.AdvancedPlots as ap
import MetBot.RainStats as rs
import MetBot.MetBlobs as blb
import MetBot.mytools as my
import MetBot.mynetcdf as mync
import glob, socket, os
import mpl_toolkits.basemap as bm
import MetBot.dset_dict as dsetdict
import MetBot.find_saddle as fs
import dsets_mplot_5group_4plot as dset_mp



### Running options
inorder=True    # to put the bars in order of number of TTTs
numlab=False     # to include number of TTTs in the yaxis label
threshtest=True # Option to run on thresholds + and - 5Wm2 as a test
peryear=True    # to divide by number of years
group=True
wh_count = 'event'  # blob or event

### Directory
bkdir=cwd+"/../../CTdata/"
botdir=bkdir+"metbot_multi_dset/"
thisdir=bkdir+"hpaperplot/"


figdir=thisdir+"barcharts_5group/"
my.mkdir_p(figdir)


### Loop threshs
if threshtest:
    thnames=['lower','actual','upper']
else:
    thnames=['actual']


### Dsets
dsets='all'
dsetnames=['noaa','cmip5']
ndset=len(dsetnames)
ndstr=str(ndset)

### Count total number of models
nm_dset=np.zeros(ndset)
for d in range(ndset):
    dset=dsetnames[d]
    nmod=len(dset_mp.dset_deets[dset])
    nm_dset[d]=nmod
nallmod=np.sum(nm_dset)
nallmod=int(nallmod)
print nallmod


if group:
    grcls=['fuchsia','gold','darkblue','r','blueviolet','springgreen']

nthresh=len(thnames)
for t in range(nthresh):

    ### Open arrays for results
    ttt_count=np.zeros(nallmod)
    cols=["" for x in range(nallmod)]
    modnm=["" for x in range(nallmod)] # creates a list of strings for modnames
    cnt = 0

    print "Looping datasets"
    for d in range(ndset):
        dset=dsetnames[d]
        dcnt=str(d+1)
        print 'Running on '+dset
        print 'This is dset '+dcnt+' of '+ndstr+' in list'

        ### Models
        mods = 'all'
        nmod = len(dset_mp.dset_deets[dset])
        mnames = list(dset_mp.dset_deets[dset])
        nmstr = str(nmod)

        for mo in range(nmod):
            name = mnames[mo]
            mcnt = str(mo + 1)
            print 'Running on ' + name
            print 'This is model ' + mcnt + ' of ' + nmstr + ' in list'

            ### if group get group
            if group:
                groupdct = dset_mp.dset_deets[dset][name]
                thisgroup = int(groupdct['group'])
                grcl = grcls[thisgroup - 1]
                cols[cnt]=grcl

            ### Find location of synop file
            outdir=botdir+dset+"/"+name+"/"
            outsuf=outdir+name+'_'

            ### Get thresh
            with open(botdir+'thresholds.fmin.all_dset.txt') as f:
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

            if wh_count == 'event':

                ###  Open synop file
                syfile=outsuf+thre_str+'_'+dset+'-OLR.synop'
                s = sy.SynopticEvents((),[syfile],COL=False)
                refkey = '0'
                key = dset + '-olr-0-' + refkey

                ### Count number of events
                ks = s.events.keys() # all events
                kw, ke = stats.spatialsubset(s,False,cutlon=40.) # splitting tracks west and east of 40E

                ### Put n events into array
                print name
                ttt_count[cnt]=len(ks)

                if peryear:

                    edts = []
                    for k in ks:
                        e = s.events[k]
                        dts = s.blobs[key]['mbt'][e.ixflags]
                        if len(dts) > 1:
                            dt = dts[len(dts) / 2]
                        else:
                            dt = dts[0]
                        edts.append(dt)
                    edts = np.asarray(edts)
                    yrs = np.unique(edts[:, 0])
                    nys=float(len(yrs))

                    print 'Dividing by number of years'
                    print nys

                    ttt_count[cnt]=ttt_count[cnt]/nys

                    print ttt_count[cnt]

            elif wh_count=='blob':

                mbsfile = outsuf + thre_str + '_' + dset + "-olr-0-0.mbs"
                refmbs, refmbt, refch = blb.mbopen(mbsfile)
                count_all = len(refmbt)

                blob_edts = []
                for b in range(len(refmbt)):
                    date = refmbt[b]
                    mon = int(date[1])
                    cX = refmbs[b, 3]
                    cY = refmbs[b, 4]

                    blob_edts.append(date)

                blob_edts = np.asarray(blob_edts)

                ttt_count[cnt] = len(blob_edts)

                if peryear:
                    yrs = np.unique(blob_edts[:, 0])
                    nys=len(yrs)

                    print 'Dividing by number of years'
                    print nys

                    ttt_count[cnt]=ttt_count[cnt]/nys

                    print ttt_count[cnt]

            ### Put name into string list
            modnm[cnt]=name
            cnt+=1

    figsuf=wh_count+'.'

    if peryear:
        figsuf=figsuf+'peryear.'

    val=ttt_count[:]

    # Open text file for results
    file = open(figdir+"nTTT_list."+thre_str+".txt", "w")

    if inorder:
        indsort=np.argsort(val)
        val4plot=val[indsort]
        mod4plot=[modnm[i] for i in indsort]
        col4plot=[cols[i] for i in indsort]
    else:
        val4plot=val
        mod4plot=modnm
        col4plot=cols

    pos=np.arange(nallmod)+0.5

    modlabels=["" for x in range(nallmod)]
    for m in range(nallmod):
        tmp=int(val4plot[m])
        strval=str(tmp)
        if numlab:
            modlabels[m]=mod4plot[m]+' ('+strval+')'
        else:
            modlabels[m] = mod4plot[m]
        file.write(mod4plot[m]+"\t"+str(int(val4plot[m]))+"\n")


    plt.figure()
    plt.subplots_adjust(left=0.3,right=0.9,top=0.9,bottom=0.1)
    plt.barh(pos,val4plot, color=col4plot, edgecolor=col4plot, align='center')
    plt.ylim(0,nallmod)
    plt.yticks(pos,modlabels,fontsize=10,fontweight='demibold')
    plt.xticks(np.arange(0,180,20),fontsize=10,fontweight='demibold')
    plt.xlabel('Number of Events',fontsize=12,fontweight='demibold')
    barfig=figdir+'/neventsbar.thresh_'+thnames[t]+'.'+figsuf+'png'
    plt.savefig(barfig,dpi=150)
    file.close()