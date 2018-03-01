"""This script creates all the ccd statistics including extinction corrected depth histograms in the observing strategy paper

It takes as input one '*-zpt.fits' file for each camera, and that table has AcoEBV column
No ccd cuts should have been applied to the zpt table, these will be applied in the script
This script does not rely on tractor,astrometry.net, etc. If you need to add some processing that requires one of these, add it to the legacy_zeropoints_merged.py script instead
"""

#if __name__ == "__main__":
#    import matplotlib
#    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
from scipy.stats import sigmaclip

from astrometry.util.fits import fits_table

from legacyzpts.qa import params 
from legacyzpts.qa.params import band2color,col2plotname

CAMERAS=['decam','mosaic','bass']
mygray='0.6'

def myhist2D(ax,x,y,xlim=(),ylim=(),nbins=()):
    #http://www.astroml.org/book_figures/chapter1/fig_S82_hess.html
    H, xbins, ybins = np.histogram2d(x,y,
                                     bins=(np.linspace(xlim[0],xlim[1],nbins[0]),
                                           np.linspace(ylim[0],ylim[1],nbins[1])))
    # Create a black and white color map where bad data (NaNs) are white
    cmap = plt.cm.binary
    cmap.set_bad('w', 1.)
    
    H[H == 0] = 1  # prevent warnings in log10
    ax.imshow(np.log10(H).T, origin='lower',
              extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              cmap=cmap, interpolation='nearest',
              aspect='auto')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def myhist(ax,data,bins=20,color='b',normed=False,ls='-',label=None):
    if label:
        _=ax.hist(data,bins=bins,facecolor=color,normed=normed,
                  histtype='stepfilled',edgecolor='none', alpha=0.75,label=label)
    else:
        _=ax.hist(data,bins=bins,facecolor=color,normed=normed,
                  histtype='stepfilled',edgecolor='none', alpha=0.75)

def myhist_step(ax,data,bins=20,color='b',normed=False,lw=2,ls='solid',label=None,
                return_vals=False):
    if label:
        h,bins,_=ax.hist(data,bins=bins,color=color,normed=normed,
                  histtype='step',lw=lw,ls=ls,label=label)
    else:
        h,bins,_=ax.hist(data,bins=bins,color=color,normed=normed,
                  histtype='step',lw=lw,ls=ls)
    if return_vals:
        return h,bins

def myscatter(ax,x,y, color='b',m='o',s=10.,alpha=0.75,label=None):
    if label is None or label == 'None':
        ax.scatter(x,y, c=color,edgecolors='none',marker=m,s=s,rasterized=True,alpha=alpha)
    else:
        ax.scatter(x,y, c=color,edgecolors='none',marker=m,s=s,rasterized=True,alpha=alpha,label=label)

def myscatter_open(ax,x,y, color='b',m='o',s=10.,alpha=0.75,label=None):
    if label is None or label == 'None':
        ax.scatter(x,y, facecolors='none',edgecolors=color,marker=m,s=s,rasterized=True,alpha=alpha)
    else:
        ax.scatter(x,y, facecolors='none',edgecolors=color,marker=m,s=s,rasterized=True,alpha=alpha,label=label)

def myannot(ax,xarr,yarr,sarr, ha='left',va='bottom',fontsize=20):
    '''x,y,s are iterable'''
    for x,y,s in zip(xarr,yarr,sarr):
        ax.annotate(s,xy=(x,y),xytext=(x,y),
                    horizontalalignment=ha,verticalalignment=va,fontsize=fontsize)

def mytext(ax,x,y,text, ha='left',va='center',fontsize=20,rotation=0,
           dataCoords=False):
    '''adds text in x,y units of fraction axis'''
    if dataCoords:
        ax.text(x,y,text, horizontalalignment=ha,verticalalignment=va,
                fontsize=fontsize,rotation=rotation)
    else:
        ax.text(x,y,text, horizontalalignment=ha,verticalalignment=va,
                transform=ax.transAxes,fontsize=fontsize,rotation=rotation)


def myerrorbar(ax,x,y, yerr=None,xerr=None,color='b',ls='none',m='o',s=10.,mew=1,alpha=0.75,label=None):
    if label is None or label == 'None':
        ax.errorbar(x,y, xerr=xerr,yerr=yerr,ls=ls,alpha=alpha,
                    marker=m,ms=s,mfc=color,mec=color,ecolor=color,mew=mew)
    else:
        ax.errorbar(x,y, xerr=xerr,yerr=yerr,ls=ls,alpha=alpha,label=label,
                    marker=m,ms=s,mfc=color,mec=color,ecolor=color,mew=mew)
        #ax.errorbar(x,y, xerr=xerr,yerr=yerr, c=color,ecolor=color,marker=m,s=s,rasterized=True,alpha=alpha,label=label)

#######
# Master depth calculator, uses non galdepth columns to get exactly galdepth in annotated-ccds
#######

class Depth(object):
    def __init__(self,camera,skyrms,gain,fwhm,zpt):
        '''
        See: observing_paper/depth.py
        Return depth that agrees with galdepth doing equiv with annotated ccd cols
        skyrms -- e/s
        gain -- e/ADU
        fwhm -- pixels so that gaussian std in pixels = fwhm/2.35
        zpt -- e/s
        '''
        assert(camera in ['decam','mosaic'])
        self.camera = camera
        self.skyrms= skyrms.copy() #Will modify to get in right units
        self.gain= gain.copy()
        self.fwhm= fwhm.copy()
        self.zpt= zpt.copy()
        if self.camera == 'decam':
            # natural camera units ADU/sec
            self.skyrms /= self.gain # e/sec --> ADU/sec
            self.zpt -= 2.5*np.log10(self.gain) # e/sec --> ADU/sec
        elif self.camera == 'mosaic':
            # e/sec are the natual camera units
            pass
        # self.get_depth_legacy_zpts(which='gal')

    def get_depth_legacy_zpts(self,which=None,func_neff=None):
        """if what to correct Neff, pass that function as arg"""
        assert(which in ['gal','psf'])
        self.which= which
        sigma= self.sigma_legacy_zpts(func_neff=func_neff) 
        return -2.5*np.log10(5 * sigma) + self.zpt #natural camera units

    def sigma_legacy_zpts(self,func_neff=None):
        if func_neff:
            neff= func_neff(self.fwhm/2.35, self.camera,self.which)
        else:
            if self.which == 'psf':
                rhalf=0
            else:
                rhalf=0.45
            neff= NeffFormulas().neff_15(self.fwhm/2.35,rhalf=rhalf)
        return self.skyrms *np.sqrt(neff)


class DepthRequirements(object):
    def __init__(self):
        self.desi= self.depth_requirement_dict()
    
    def get_single_pass_depth(self,band,which,camera):
        assert(which in ['gal','psf'])
        assert(camera in ['decam','mosaic'])
        # After 1 pass
        if camera == 'decam':
            return self.desi[which][band] - 2.5*np.log10(np.sqrt(2))
        elif camera == 'mosaic':
            return self.desi[which][band] - 2.5*np.log10(np.sqrt(3))

    def depth_requirement_dict(self):
        mags= defaultdict(lambda: defaultdict(dict))
        mags['gal']['g']= 24.0
        mags['gal']['r']= 23.4
        mags['gal']['z']= 22.5
        mags['psf']['g']= 24.7
        mags['psf']['r']= 23.9
        mags['psf']['z']= 23.0
        return mags


def std_err_mean(x):
    return np.std(x)/np.sqrt(len(x)-1)

#######
# Histograms of CCD statistics from legacy zeropoints
# for paper
#######
# a= ZeropointHistograms('/path/to/combined/decam-zpt.fits','path/to/mosaics-zpt.fits')

def err_on_radecoff(rarms,decrms,nmatch):
    err_raoff= 1.253 * rarms / np.sqrt(nmatch)
    err_decoff= 1.253 * decrms / np.sqrt(nmatch)
    return np.sqrt(err_raoff**2 + err_decoff**2)


def fitfunc_raoff_v_mjd(p,x):
    return p[0]*np.cos(2*np.pi/p[1]*x+p[2]) + p[3] 

class ZeropointHistograms(object):
    '''Histograms for papers'''
    def __init__(self,decam=None,mosaic=None,bass=None):
        self.decam= decam
        self.mosaic= mosaic
        self.bass= bass
        if self.decam:
            self.decam= fits_table(self.decam)
        if self.mosaic:
            self.mosaic= fits_table(self.mosaic)
        if self.bass:
            self.bass= fits_table(self.bass)
        self.clean()
        self.mjd_sorted_order()
        self.add_keys()
        self.num_exp= self.get_num_exp()

    def mjd_sorted_order(self):
        if self.decam:
            isort= np.argsort(self.decam.mjd_obs)
            self.decam= self.decam[isort]
        if self.mosaic:
            isort= np.argsort(self.mosaic.mjd_obs)
            self.mosaic= self.mosaic[isort]
        if self.bass:
            isort= np.argsort(self.bass.mjd_obs)
            self.bass= self.bass[isort]
    
    def plot(self):
        self.plot_hist_1d_perband()
        self.plot_hist_1d()
        #[('rarms','decrms'),('raoff','decoff'),('phrms','phoff')]
        self.seaborn_contours(x_key='raoff',y_key='decoff')
        self.plot_astro_photo_scatter()
        self.plot_hist_depth(legend=False)
        self.print_ccds_table()
        self.print_requirements_table()
    
    def plot_v_mjd(self,camera):
        if not hasattr(self,camera+'_avg'):
            self.group_by_actualDateObs(camera)
        self.plot_zpt_v_mjd(camera)
        self.plot_raoff_v_mjd_scaled(camera)
        self.plot_radecoff_v_mjd(camera)

    def plot_zpt_bimodality(self,camera):
        self.Z.plot_v_mjd('decam')
    
    def group_by_actualDateObs(self,camera):
        cols= ["actualDateObs","mjd_obs",'seeing','zpt','transp','skymag','airmass',
               "decoff","raoff","filter",
               'skycounts','skyrms']
        df= pd.DataFrame({col:getattr(self,camera).get(col) 
                          for col in cols})
        df= df.apply(lambda x: x.values.byteswap().newbyteorder())
        setattr(self,camera+'_avg',
                df.groupby(['actualDateObs']).agg(np.mean).reset_index())
        setattr(self,camera+'_stderr',
                df.groupby(['actualDateObs']).agg(std_err_mean).reset_index())
    
    def clean(self):
        if self.decam:
            print('cleaning decam')
            self._clean(self.decam)
        if self.mosaic:
            print('cleaning mosaic')
            self._clean(self.mosaic)
        if self.bass:
            print('cleaning bass')
            self._clean(self.bass)

    def _clean(self,T):
        if 'err_message' in T.get_columns():
            keep= (pd.Series(T.get('err_message')).str.strip().str.len() == 0).values
            if len(T[keep]) < len(T):
                T.cut(keep)
                print('Cutting err_message to %d/%d' % (len(T),len(keep)))
        # Nans
        keep=np.ones(len(T),dtype=bool)
        for key in ['zpt','fwhm']:
            keep*= np.isfinite(T.get(key))
        if len(T[keep]) < len(T):
            T.cut(keep)
            print('Cutting finite to %d/%d' % (len(T),len(keep)))
        # Negatives
        for key in ['zpt','airmass']:
            keep= T.get(key) > 0 
            if len(T[keep]) < len(T):
                T.cut(keep)
                print('Cutting %s > 0 to %d/%d' % (key,len(T),len(keep)))
        # grz only
        isGrz= pd.Series(T.get('filter')).str.strip().isin(['g','r','z']).values
        if len(T[isGrz]) < len(T):
            T.cut(isGrz)
            print('Cutting to grz only %d/%d' % (len(T),len(isGrz)))
            

    def add_keys(self):
        if self.decam:
            self.add_keys_numeric(self.decam,'decam')
            self.add_keys_actualDate(self.decam,'decam')
        if self.mosaic:
            self.add_keys_numeric(self.mosaic,'mosaic')
            self.add_keys_actualDate(self.mosaic,'mosaic')
        if self.bass:
            self.add_keys_numeric(self.bass,'bass')
            self.add_keys_actualDate(self.bass,'bass')

    def add_keys_numeric(self,T,camera=None):
        assert(camera in CAMERAS)
        Aco= {'decam':dict(g=3.214,r=2.165,z=1.562),
              'mosaic':dict(z=1.562),
              'bass':dict(g=3.214,r=2.165)}
        Pix= {'decam':0.262,
              'mosaic':0.26,
              'bass':0.455}
        data= np.zeros(len(T))+np.nan
        for band in Aco[camera].keys():
            keep= T.filter == band
            if len(T[keep]) > 0:
                data[keep]= Aco[camera][band] * T.ebv[keep]
        T.set('AcoEBV',data)
        T.set('seeing',T.fwhm * Pix[camera])
        try:
            nmatch= T.nmatch_photom
        except AttributeError:
            nmatch= T.nmatch
        T.set('err_on_phoff',1.253 * T.phrms/np.sqrt(nmatch)) # error on median
        T.set('err_on_radecoff',err_on_radecoff(T.rarms,T.decrms,nmatch))
        # Depths
        depth_obj= Depth(camera, T.skyrms,T.gain,
                         T.fwhm,T.zpt)
        T.set('psfdepth', depth_obj.get_depth_legacy_zpts('psf'))
        T.set('galdepth', depth_obj.get_depth_legacy_zpts('gal'))
        for col in ['psfdepth','galdepth']:
            T.set(col+'_extcorr', T.get(col) - T.AcoEBV)
 
    def add_keys_actualDate(self,T,camera=None):
        if camera == 'decam':
            self._add_keys_actualDate(T,threshold='15:00:00.0')
        elif camera == 'mosaic':
            T.set('actualDateObs',T.date_obs)
        else:
            raise ValueError('bass actualDate figured out yet')
    
    def _add_keys_actualDate(self,T,threshold='15:00:00.0'):
        actualDateObs= np.array(['0000-00-00']*len(T))
        sameNight= T.ut < threshold
        for yyyymmdd in set(T.date_obs):
            thisNight= (T.date_obs == yyyymmdd) & sameNight
            nextNight= (T.date_obs == yyyymmdd) & ~sameNight
            TM= pd.Timestamp(yyyymmdd)
            actualDateObs[thisNight]= TM.strftime('%Y-%m-%d')
            actualDateObs[nextNight]= (TM + pd.DateOffset(1)).strftime('%Y-%m-%d') 
        assert(len(actualDateObs[actualDateObs == '0000-00-00']) == 0)
        T.set('actualDateObs',actualDateObs)
    
    def get_num_exp(self):
        num={}
        if self.mosaic:
            num['mosaic_z']= len(set(self.mosaic.expnum))
        if self.decam:
            for band in ['g','r','z']:
                keep= self.decam.filter == band
                num['decam_%s' % band]= len(set(self.decam.expnum[keep]))
        if self.bass:
            for band in ['g','r']:
                keep= self.bass.filter == band
                num['bass_%s' % band]= len(set(self.bass.expnum[keep]))
        return num

    def get_numeric_keys(self):
        keys= \
            ['skymag','skyrms','zpt','airmass','fwhm','gain','transp',
             'seeing','psfdepth','galdepth']
        return keys

    def get_defaultdict_ylim(self,ylim=None):
        ylim_dict=defaultdict(lambda: ylim)
        ylim_dict['skymag']= (0,2.)
        ylim_dict['zpt']= (0,5.)
        ylim_dict['transp']= (0,9)
        return ylim_dict

    def get_defaultdict_xlim(self,xlim=None):
        xlim_dict=defaultdict(lambda: xlim)
        xlim_dict['skymag']= (17,22.5)
        xlim_dict['skyrms']= (0,1.5)
        xlim_dict['zpt']= (25.8,27.2)
        xlim_dict['airmass']= (0.9,2.5)
        xlim_dict['transp']= (0.5,1.4)
        xlim_dict['gain']= (0,5)
        xlim_dict['seeing']= (0.5,2.4)
        xlim_dict['fwhm']= (2,10)
        xlim_dict['psfdepth']= (21,25)
        xlim_dict['galdepth']= (21,25)
        return xlim_dict      

    def get_fiducial(self,key,camera,band):
        assert(camera in ['decam','mosaic'])
        d= {}
        X0, see0= 1.3, 1.3
        if camera == 'decam':
            d['zpt']= dict(g=26.610, r=26.818, z=26.484) # e/s
            d['skymag']= dict(g=22.04, r=20.91, z=18.46)
            d['airmass']= dict(g=X0,r=X0,z=X0) #arcsec
            d['seeing']= dict(g=see0,r=see0,z=see0) #arcsec
        elif camera == 'mosaic':
            d['zpt']= dict(z=26.552) # e/s
            d['skymag']= dict(z=18.46)
            d['airmass']= dict(z=X0) #arcsec
            d['seeing']= dict(z=see0) #arcsec
        return d[key][band]

    def plot_zpt_v_mjd(self,camera):
        import seaborn as sns
        g = sns.JointGrid(x='mjd_obs',y="zpt", data=getattr(self,camera+'_avg'))

        kwargs= dict(ls='none',alpha=0.5,marker='o',ms=5,
                     mfc='b',mec='b',ecolor='g',mew=1,capsize=3)
        g = g.plot_joint(plt.errorbar, yerr=getattr(self,camera+'_stderr')['zpt'], **kwargs)

        params= LeastSquares(getattr(self,camera+'_avg')['mjd_obs'],
                             getattr(self,camera+'_avg')['zpt'],
                             getattr(self,camera+'_stderr')['zpt']).estim()
        x= np.linspace(getattr(self,camera+'_avg')['mjd_obs'].min(),
                       getattr(self,camera+'_avg')['mjd_obs'].max())
        g.ax_joint.plot(x,params['slope']*x+params['inter'],'r--',lw=2,
                        label=r'$%.2g \, \rm{MJD} + %.2g$' % (params['slope'],params['inter']))
        g.ax_joint.legend(loc='upper right')

        g = g.plot_marginals(sns.distplot, kde=False, color=kwargs['mfc'])

        g.ax_joint.set_ylim(26,27)

        savefn='zpt_errorbars_%s.png' % camera
        g.savefig(savefn, bbox_inches='tight',dpi=150)
        print("wrote %s" % savefn)

    def fit_raoff_v_mjd_scaled(self,camera):
        # Scale
        scalers= dict(min_mjd= np.min(getattr(self,camera+'_avg')['mjd_obs']),
                      scale_mjd=3.5,scale_raoff=5)
        x= getattr(self,camera+'_avg')['mjd_obs'] - scalers['min_mjd']
        scalers.update(max_mjd= np.max(x))
        x= x/scalers['max_mjd'] * scalers['scale_mjd']
        y= getattr(self,camera+'_avg')['raoff'] * scalers['scale_raoff']
        keep= x <= 2.4
        x= x[keep]
        y= y[keep]
        # Fit
        from scipy.optimize import leastsq
        #fitfunc = lambda p, x: p[0]*np.cos(2*np.pi/p[1]*x+p[2]) + p[3] 
        errfunc = lambda p, x, y: fitfunc_raoff_v_mjd(p, x) - y 
        p0 = [1., 1., 0., 0.] # Initial guess
        p1, success = leastsq(errfunc, p0[:], args=(x, y))
        assert(success != 0)
        return p1,scalers,x,y
 
    def plot_raoff_v_mjd_scaled(self,camera):
        p1,_,x,y= self.fit_raoff_v_mjd_scaled(camera)

        import seaborn as sns
        g= sns.jointplot(x,y)
        new_x= np.linspace(x.min(),x.max(),num=100)
        g.ax_joint.plot(new_x,fitfunc_raoff_v_mjd(p1,new_x),'k--',lw=2)
        #g.ax_joint.plot(new_x,fitfunc(p0,new_x),'g--',lw=2)
        xlab=plt.xlabel('MJD (scaled)')
        ylab=plt.ylabel('raoff (scaled)')
        savefn='radecoff_v_mjd_%s_scaled.png' % camera
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
        print("wrote %s" % savefn)

    def plot_radecoff_v_mjd(self,camera):
        p1,scalers,x,y= self.fit_raoff_v_mjd_scaled(camera)
        # Convert from scaled to actual coords
        amp= p1[0]/scalers['scale_raoff']
        mjd_period= p1[1] * scalers['max_mjd'] / scalers['scale_mjd']
        phase_shift= -p1[2]/mjd_period
        offset= p1[3]/scalers['scale_raoff']

        p_new= [amp,mjd_period,phase_shift,offset]
        # Plot
        import seaborn as sns
        fig,ax=plt.subplots(2,1,figsize=(6,10))
        plt.subplots_adjust(hspace=0)

        kwargs= dict(ls='none',alpha=0.2,marker='o',ms=5,
                     mfc='b',mec='b',ecolor='g',mew=1,capsize=3)
        for row,Y in zip(range(2),["raoff","decoff"]):
            ax[row].errorbar(getattr(self,camera+'_avg')["mjd_obs"],
                             getattr(self,camera+'_avg')[Y], 
                             yerr=getattr(self,camera+'_stderr')[Y], **kwargs)
            if Y == 'raoff':
                new_x= np.linspace(getattr(self,camera+'_avg')['mjd_obs'].min(),
                                   getattr(self,camera+'_avg')['mjd_obs'].max(),num=100)
                ax[row].plot(new_x,fitfunc_raoff_v_mjd(p_new,new_x),'r-',lw=2,
                             label=r'$%.2f \, \cos\left(2\pi \, \rm{MJD} \, / \, %.1f + %.2f\right) + %.2f$' % \
                                   (p_new[0],p_new[1],p_new[2],p_new[3]))
            if Y == "raoff":
                ax[row].set_ylim(-0.2,0.3)
                handles, labels = ax[row].get_legend_handles_labels()
                ax[row].legend([handles[0]],[labels[0]],loc=(0.12,0.90),fontsize=12)
            else:
                ax[row].set_ylim(-0.25,0.05)
            ylab= ax[row].set_ylabel(Y)
        xlab= ax[1].set_xlabel('MJD')

        savefn='radecoff_v_mjd_%s.png' % camera
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
        print("wrote %s" % savefn)



    def plot_hist_1d(self):
        # All keys and any ylims to use
        cols= self.get_numeric_keys()
        ylim= self.get_defaultdict_ylim()
        xlim= self.get_defaultdict_xlim()
        fiducial_keys= ['seeing','zpt','skymag','airmass']
        # Plot
        FS=14
        eFS=FS+5
        tickFS=FS
        for key in cols:
            fig,ax= plt.subplots(figsize=(7,5))
            if xlim[key]:
                bins= np.linspace(xlim[key][0],xlim[key][1],num=40)
            else:
                bins=40
            # decam
            if self.decam:
                for band in set(self.decam.filter):
                    keep= self.decam.filter == band
                    print(key,set(self.decam.filter),band,len(self.decam[keep]))
                    myhist_step(ax,self.decam.get(key)[keep], bins=bins,normed=True,
                                color=band2color(band),ls='solid',
                                label='%s (DECam, %d)' % (band,self.num_exp['decam_'+band]))
                    if key in fiducial_keys:
                        ax.axvline(self.get_fiducial(key,'decam',band),
                                   c=band2color(band),ls='dotted',lw=1) 
                    
            # mosaic
            if self.mosaic:
                for band in set(self.mosaic.filter):
                    keep= self.mosaic.filter == band
                    myhist_step(ax,self.mosaic.get(key)[keep], bins=bins,normed=True,
                                color=band2color(band),ls='dashed',
                                label='%s (Mosaic3, %d)' % (band,self.num_exp['mosaic_'+band]))
                    if key in fiducial_keys:
                        ls= 'dotted'
                        if key == 'zpt':
                            ls= 'dashed'
                        ax.axvline(self.get_fiducial(key,'mosaic',band),
                                   c=band2color(band),ls=ls,lw=1) 
            # Label
            ylab=ax.set_ylabel('PDF',fontsize=FS)
            ax.tick_params(axis='both', labelsize=tickFS)
            leg=ax.legend(loc=(0,1.02),ncol=2,fontsize=FS-2)
            if ylim[key]:
                ax.set_ylim(ylim[key])
            if xlim[key]:
                ax.set_xlim(xlim[key])
            xlab=ax.set_xlabel(col2plotname(key),fontsize=FS) #0.45'' galaxy
            savefn='hist_1d_%s.png' % key
            plt.savefig(savefn, bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight')
            plt.close() 
            print("wrote %s" % savefn)

    def plot_hist_1d_perband(self):
        obs_programs= pd.Series(self.decam.image_filename).str.split('/').str[1]
        set_programs= set(obs_programs) 
        print('set_programs=',set_programs)
        raise ValueError
        # All keys and any ylims to use
        cols= self.get_numeric_keys()
        ylim= self.get_defaultdict_ylim()
        xlim= self.get_defaultdict_xlim()
        fiducial_keys= ['seeing','zpt','skymag','airmass']
        # Plot
        FS=14
        eFS=FS+5
        tickFS=FS
        for key in cols:
            fig,ax= plt.subplots(figsize=(7,5))
            if xlim[key]:
                bins= np.linspace(xlim[key][0],xlim[key][1],num=40)
            else:
                bins=40
            # decam
            if self.decam:
                for band in set(self.decam.filter):
                    keep= self.decam.filter == band
                    print(key,set(self.decam.filter),band,len(self.decam[keep]))
                    myhist_step(ax,self.decam.get(key)[keep], bins=bins,normed=True,
                                color=band2color(band),ls='solid',
                                label='%s (DECam, %d)' % (band,self.num_exp['decam_'+band]))
                    if key in fiducial_keys:
                        ax.axvline(self.get_fiducial(key,'decam',band),
                                   c=band2color(band),ls='dotted',lw=1) 
                    
            # mosaic
            if self.mosaic:
                for band in set(self.mosaic.filter):
                    keep= self.mosaic.filter == band
                    myhist_step(ax,self.mosaic.get(key)[keep], bins=bins,normed=True,
                                color=band2color(band),ls='dashed',
                                label='%s (Mosaic3, %d)' % (band,self.num_exp['mosaic_'+band]))
                    if key in fiducial_keys:
                        ls= 'dotted'
                        if key == 'zpt':
                            ls= 'dashed'
                        ax.axvline(self.get_fiducial(key,'mosaic',band),
                                   c=band2color(band),ls=ls,lw=1) 
            # Label
            ylab=ax.set_ylabel('PDF',fontsize=FS)
            ax.tick_params(axis='both', labelsize=tickFS)
            leg=ax.legend(loc=(0,1.02),ncol=2,fontsize=FS-2)
            if ylim[key]:
                ax.set_ylim(ylim[key])
            if xlim[key]:
                ax.set_xlim(xlim[key])
            xlab=ax.set_xlabel(col2plotname(key),fontsize=FS) #0.45'' galaxy
            savefn='hist_1d_%s.png' % key
            plt.savefig(savefn, bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight')
            plt.close() 
            print("wrote %s" % savefn)



    def plot_hist_depth(self,legend=True):
        # All keys and any ylims to use
        cols= ['psfdepth_extcorr','galdepth_extcorr']
        xlim= (21,24.5)
        ylim= None
        # Depths
        depth_obj= DepthRequirements()
        # Plot
        FS=14
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(2,2,figsize=(10,8))
        if legend:
            plt.subplots_adjust(hspace=0.2,wspace=0)
        else:
            plt.subplots_adjust(hspace=0,wspace=0)
        if xlim:
            bins= np.linspace(xlim[0],xlim[1],num=40)
        else:
            bins=40
        d_row=dict(decam=0,mosaic=1,bok=1)
        d_col=dict(psf=0,gal=1)
        for which in ['psf','gal']:
            key= which+'depth_extcorr'
            # decam
            if self.decam:
                row=d_row['decam']
                col=d_col[which]
                for band in set(self.decam.filter):
                    keep= self.decam.filter == band
                    myh, mybins= myhist_step(ax[row,col],self.decam.get(key)[keep], bins=bins,normed=True,
                                color=band2color(band),ls='solid',lw=2,
                                label='%s (DECam)' % band, return_vals=True)
                    # requirements
                    if which == 'gal':
                        p1_depth= depth_obj.get_single_pass_depth(band,which,'decam')
                        ax[row,col].axvline(p1_depth,
                                        c=band2color(band),ls='dashed',lw=1,
                                        label=r'$\mathbf{m_{\rm{DESI}}}$= %.2f' % p1_depth)
                        # 90% > than requirement
                        q10= np.percentile(self.decam.get(key)[keep], q=10)
                        #ax[row,col].axvline(q10,
                        #                c=band2color(band),ls='dotted',lw=2,
                        #                label='q10= %.2f' % q10)
                        mybins= mybins[:-1]
                        lasth= myh[mybins <= q10][-1]
                        myh= np.append(myh[mybins <= q10], lasth)
                        mybins= np.append(mybins[mybins <= q10], q10)
                        ax[row,col].fill_between(mybins,[0]*len(myh),myh,
                                                 where= mybins <= q10, interpolate=True,step='post',
                                                 color=band2color(band),alpha=0.5)
            # mosaic
            if self.mosaic:
                row=d_row['mosaic']
                col=d_col[which]
                for band in set(self.mosaic.filter):
                    mos_color='k'
                    keep= self.mosaic.filter == band
                    myh, mybins= myhist_step(ax[row,col],self.mosaic.get(key)[keep], bins=bins,normed=True,
                                color=band2color(band),ls='solid',lw=2,
                                label='%s (Mosaic3)' % band, return_vals=True)
                    # requirements
                    if which == 'gal':
                        p1_depth= depth_obj.get_single_pass_depth(band,which,'mosaic')
                        ax[row,col].axvline(p1_depth,
                                        c=band2color(band),ls='dashed',lw=1,
                                        label=r'$\mathbf{m_{\rm{DESI}}}$= %.2f' % p1_depth)
                        # 90% > than requirement
                        q10= np.percentile(self.mosaic.get(key)[keep], q=10)
                        #ax[row].axvline(q10,
                        #                c=mos_color,ls='dotted',lw=2,
                        #                label='q10= %.2f' % q10)
                        mybins= mybins[:-1]
                        lasth= myh[mybins <= q10][-1]
                        myh= np.append(myh[mybins <= q10], lasth)
                        mybins= np.append(mybins[mybins <= q10], q10)
                        ax[row,col].fill_between(mybins,[0]*len(myh),myh,
                                                 where= mybins <= q10, interpolate=True,step='post',
                                                 color=band2color(band),alpha=0.5)
        # Label
        cam={}
        cam['0']='DECaLS'
        cam['1']='MzLS'
        for row in [1,0]:
            for col in [0,1]:
                if legend:
                    leg=ax[row,col].legend(loc=(0.,1.02),ncol=3,fontsize=FS-5)
                ax[row,col].tick_params(axis='both', labelsize=tickFS)
                if ylim:
                    ax[row,col].set_ylim(ylim)
                if xlim:
                    ax[row,col].set_xlim(xlim)
            ylab=ax[row,0].set_ylabel('PDF (%s)' % cam[str(row)],fontsize=FS)
        for col,which in zip([0,1],['psf','gal']): 
            key= which+'depth_extcorr'
            xlab=ax[1,col].set_xlabel(r'%s' % col2plotname(key),fontsize=FS) 
        # Hide axis labels
        for col in [0,1]: 
            ax[0,col].set_xticklabels([])
        for row in [0,1]:
            ax[row,1].set_yticklabels([])
        savefn='hist_depth.png'
        bbox=[xlab,ylab]
        if legend:
            bbox.append(leg)
        plt.savefig(savefn, bbox_extra_artists=bbox, bbox_inches='tight')
        plt.close() 
        print("wrote %s" % savefn)


    def get_lim(self,col):
        d= dict(rarms= (0,0.5),
                decrms= (0,0.5),
                phrms= (0,0.5),
                raoff= dict(decam=(-0.25,0.25),
                            mosaic=(-0.05,0.05)),
                decoff= dict(decam=(-0.25,0.1),
                             mosaic=(-0.05,0.05)),
                phoff= (-6,6),
                err_on_phoff= (0,0.08),
                err_on_radecoff= (0,0.3))
        return d.get(col,None)

    def seaborn_contours(self,x_key='raoff',y_key='decoff'):
        #import seaborn as sns
        nbins=dict(decam=(40,40),mosaic=(40,40))
        if x_key == 'raoff' and y_key == 'decoff':
            xlim=dict(decam=(-0.5,0.5),
                      mosaic=(-0.1,0.1))
            ylim=dict(decam=(-0.5,0.25),
                      mosaic=(-0.1,0.1))
        cmaps=dict(g='Greens_d',r='Reds_d',z='Blues_d')

        fig,axes= plt.subplots(2,2,figsize=(8,6))
        ax= axes.flatten()
        plt.subplots_adjust(hspace=0.15,wspace=0.2)
        for i,camera,band in [(0,'decam','g'),(1,'decam','r'),
                              (2,'decam','z'),(3,'mosaic','z')]:
            if camera == 'decam':
                x,y= self.decam.get(x_key), self.decam.get(y_key)
                keep= self.decam.filter == band
            if camera == 'mosaic':
                x,y= self.mosaic.get(x_key), self.mosaic.get(y_key)
                keep= self.mosaic.filter == band
            #sns.kdeplot(x[keep], y[keep], ax=ax[0,col],
            #            cmap=cmap[camera], shade=False, shade_lowest=False)
            myhist2D(ax[i],x[keep],y[keep],
                     xlim=xlim[camera],ylim=ylim[camera],nbins=nbins[camera])
            ax[i].text(0.05,0.95,'%s, %s' % (camera,band), 
                       horizontalalignment='left',verticalalignment='center',
                       transform=ax[i].transAxes)
            ax[i].text(0.05,0.88,'(%d)' % self.num_exp[camera+'_'+band], 
                       horizontalalignment='left',verticalalignment='center',
                       transform=ax[i].transAxes)
        # Crosshairs
        for i in range(4):
            ax[i].axhline(0,c='r',ls='--',lw=1)
            ax[i].axvline(0,c='r',ls='--',lw=1)
        # Label
        for i in [2,3]:
            xlab=ax[i].set_xlabel(col2plotname(x_key)) #0.45'' galaxy
            #ax[1,col].tick_params(axis='both')
        for i in [0,2]:
            ylab=ax[i].set_ylabel(col2plotname(y_key))
            #ax[row,0].tick_params(axis='both')
        for i in [1]:
            ax[i].yaxis.set_ticklabels([])

        savefn='densitymap_%s_%s.png' % (x_key,y_key)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
        print("wrote %s" % savefn)

    def plot_astro_photo_scatter(self,prefix=''):
        # Plot
        FS=14
        eFS=FS+5
        tickFS=FS
        x_key,y_key= 'err_on_radecoff','err_on_phoff'
        fig,ax= plt.subplots(2,1,figsize=(8,10))
        plt.subplots_adjust(hspace=0.2,wspace=0)
        # g,r band
        row=0
        if self.decam:            
            x,y= self.decam.get(x_key), self.decam.get(y_key)
            for band in ['g','r']:
                keep= self.decam.filter == band
                if np.where(keep)[0].size > 0:
                    myscatter(ax[row],x[keep],y[keep], 
                              color=band2color(band),s=10.,alpha=0.75,
                              label='%s (DECam, %d)' % (band,len(x[keep])))
        # z-band
        row=1
        if self.decam:            
            x,y= self.decam.get(x_key), self.decam.get(y_key)
            for band in ['z']:
                keep= self.decam.filter == band
                if np.where(keep)[0].size > 0:
                    myscatter(ax[row],x[keep],y[keep], 
                              color=band2color(band),s=10.,alpha=0.75,
                              label='%s (DECam, %d)' % (band,len(x[keep])))
        if self.mosaic:            
            x,y= self.mosaic.get(x_key), self.mosaic.get(y_key)
            for band in set(self.mosaic.filter):
                keep= self.mosaic.filter == band
                if np.where(keep)[0].size > 0:
                    myscatter(ax[row],x[keep],y[keep], 
                              color='k',s=10.,alpha=0.75,
                              label='%s (Mosaic3, %d)' % (band,len(x[keep])))
        # Crosshairs
        ax[0].axhline(0.01,c='k',ls='dashed',lw=1)
        ax[1].axhline(0.02,c='k',ls='dashed',lw=1)
        for row in range(2):
            ax[row].axvline(0.03,c='k',ls='dashed',lw=1)
        # Label
        xlab=ax[1].set_xlabel(col2plotname(x_key),fontsize=FS) #0.45'' galaxy
        for row in range(2):
            ylab=ax[row].set_ylabel(col2plotname(y_key),fontsize=FS)
            ax[row].tick_params(axis='both', labelsize=tickFS)
            leg=ax[row].legend(loc='upper right',scatterpoints=1,markerscale=3.,fontsize=FS)
            if self.get_lim(x_key):
                ax[row].set_xlim(self.get_lim(x_key))
            if self.get_lim(y_key):
                ax[row].set_ylim(self.get_lim(y_key))
        savefn='astro_photo_error.png'
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print("wrote %s" % savefn)

    def col2name(self,col):
        d=dict(skycounts='Sky Level',
               skyrms='Sky RMS',
               skymag='Sky Brightness',
               zpt='Zeropoint',
               seeing='Seeing',
               airmass='Airmass',
               transp='Atmospheric Transp',
               phoff='Photometric Offset',
               raoff='Ra Offset',
               decoff='Dec Offset')
        return d[col]

    def print_ccds_table(self):
        print('CCDS TABLE')
        print('statistic & units & DECam g & DECam r & DECam z & Mzls z')
        for col in ['skycounts','skyrms','skymag','zpt','seeing','airmass','transp','phoff','raoff','decoff']:
            if col in ['phoff','raoff','decoff']:
                # has an uncertainty
                # Assume we have decam
                text= '& %s & units & %.2f(%.2f) & %.2f(%.2f) & %.2f(%.2f)' % \
                        (self.col2name(col),
                             np.median( self.decam.get(col)[self.decam.filter == 'g'] ),
                                np.std( self.decam.get(col)[self.decam.filter == 'g'] ),
                             np.median( self.decam.get(col)[self.decam.filter == 'r'] ),
                                np.std( self.decam.get(col)[self.decam.filter == 'r'] ),
                             np.median( self.decam.get(col)[self.decam.filter == 'z'] ),
                                np.std( self.decam.get(col)[self.decam.filter == 'z'] )
                        )
                if self.mosaic:
                    text= text + ' & %.2f(%.2f) \\\\' % \
                            (np.median( self.mosaic.get(col)[self.mosaic.filter == 'z'] ),
                                np.std( self.mosaic.get(col)[self.mosaic.filter == 'z'] )
                            )
                else:
                    text= text + ' & --(--) \\\\' 
                print(text)
            else:
                # No error computed
                # Assume we have decam
                text= '& %s & units & %.2f & %.2f & %.2f' % \
                        (self.col2name(col),
                             np.median( self.decam.get(col)[self.decam.filter == 'g'] ),
                             np.median( self.decam.get(col)[self.decam.filter == 'r'] ),
                             np.median( self.decam.get(col)[self.decam.filter == 'z'] )
                        )
                if self.mosaic:
                    text= text +  ' & %.2f  \\\\' % \
                            (np.median( self.mosaic.get(col)[self.mosaic.filter == 'z'] ),
                            )
                else:
                    text= text +  ' & --  \\\\'  
                print(text)

    def print_requirements_table(self):
        print('DESI REQUIREMENTS TABLE')
        print('camera & band & psfdepth: q10 q50 desi_req & galdepth: q10 q50 desi_req')
        phrms= dict(g=0.01,r=0.01,z=0.02)
        depth_obj= DepthRequirements()
        need_cols= ['psfdepth_extcorr','galdepth_extcorr',
                    'err_on_radecoff','err_on_phoff']
        for band in ['g','r','z']:
            keep= (self.decam.filter == band)
            for col in need_cols:
                keep *= (np.isfinite(self.decam.get(col)))
            p1_depth_psf= depth_obj.get_single_pass_depth(band,'psf','decam')
            p1_depth_gal= depth_obj.get_single_pass_depth(band,'gal','decam')
            print('DECam & %s & %.2f | %.2f | %.2f & %.2f | %.2f | %.2f \\\\' % \
                    (band,np.percentile( self.decam.psfdepth_extcorr[keep], q=10),
                          np.percentile( self.decam.psfdepth_extcorr[keep], q=50),
                            p1_depth_psf,
                          np.percentile( self.decam.galdepth_extcorr[keep], q=10),
                          np.percentile( self.decam.galdepth_extcorr[keep], q=50),
                            p1_depth_gal,
                          #np.median( self.decam.err_on_radecoff[keep]),
                          #  0.030,
                          #np.median( self.decam.err_on_phoff[keep]),
                          #  phrms[band],
                    ))
        if self.mosaic:
            for band in ['z']:
                keep= self.mosaic.filter == band
                for col in need_cols:
                    keep *= (np.isfinite(self.mosaic.get(col)))
                p1_depth_psf= depth_obj.get_single_pass_depth(band,'psf','mosaic')
                p1_depth_gal= depth_obj.get_single_pass_depth(band,'gal','mosaic')
                print('MOSAIC-3 & %s & %.2f | %.2f | %.2f & %.2f | %.2f | %.2f \\\\' % \
                        (band,np.percentile( self.mosaic.psfdepth_extcorr[keep], q=10),
                              np.percentile( self.mosaic.psfdepth_extcorr[keep], q=50),
                                p1_depth_psf,
                              np.percentile( self.mosaic.galdepth_extcorr[keep], q=10),
                              np.percentile( self.mosaic.galdepth_extcorr[keep], q=50),
                                p1_depth_gal,
                              #np.median( self.mosaic.err_on_radecoff[keep]),
                              #  0.030,
                              #np.median( self.mosaic.err_on_phoff[keep]),
                              #  phrms[band],
                        ))





#######
# Needed exposure time calculator
#######
# compare to obsdb's value
# zp= ZptsTneed(zpt_fn='/global/cscratch1/sd/kaylanb/publications/observing_paper/neff_empirical/test_1000-zpt.fits.gz',camera=camera)
# db= LoadDB(camera=camera) 
# Match
# m= db.match_to_zeropoints(tab=zp.data)

class CalculateTneed(object):
    def calc(self,camera,filter_arr,fwhm_pix,transp,airmass,ebv,msky_ab_per_arcsec2):
        self.fid= params.get_fiducial(camera=camera)
        # Obsbot uses Neff_gal not psf
        neff_fid = self.Neff_gal(self.fid.fwhm0 / self.fid.pixscale)
        neff     = self.Neff_gal(fwhm_pix)
        
        Kco_arr= np.zeros(len(transp))-1
        Aco_arr= np.zeros(len(transp))-1
        sky0_arr= np.zeros(len(transp))-1
        t0_arr= np.zeros(len(transp))-1
        for band in self.fid.bands:
            hasband= filter_arr == band
            Kco_arr[hasband]= self.fid.Kco[band]
            Aco_arr[hasband]= self.fid.Aco[band]
            sky0_arr[hasband]= self.fid.sky0[band]
            t0_arr[hasband]= self.fid.t0[band]
        #
        scaling =  1./transp**2 *\
                   10.**(0.8 * Kco_arr * (airmass - 1.)) *\
                   10.**(0.8 * Aco_arr * ebv) *\
                   (neff / neff_fid) *\
                   10.**(-0.4 * (msky_ab_per_arcsec2 - sky0_arr))
        assert(np.all(scaling > 0.))
        return scaling * t0_arr

    def Neff_gal(self,fwhm_pix):
        r_half = 0.45 #arcsec
        # magic 2.35: convert seeing FWHM into sigmas in arcsec.
        return 4. * np.pi * (fwhm_pix / 2.35)**2 +\
               8.91 * r_half**2 +\
               self.fid.pixscale**2/12.



class ZptsTneed(object): 
    '''depths: annotated ccd vs. legacy'''
    def __init__(self,zpt_fn=None,camera=None):
        assert(camera in ['decam','mosaic'])
        self.camera = camera
        self.data= fits_table(zpt_fn)
        # process
        self.apply_cuts()
        #self.convert_units()
        self.add_cols()

    def convert_units(self):
        if self.camera == 'decam':
            self.data.set('transp', self.data.transp * 10**(-0.4* 2.5*np.log10(self.data.gain)))
            self.data.set('skymag', self.data.skymag + 2.5*np.log10(self.data.gain))
        elif self.camera == 'mosaic':
            self.data.set('transp', self.data.transp * 10**(-0.4* 2.5*np.log10(self.data.gain)))
            self.data.set('skymag', self.data.skymag + 2.5*np.log10(self.data.gain))
    
    def add_cols(self):
        # Add EBV
        sfd = SFDMap()
        self.data.set('ebv', sfd.ebv(self.data.ra, self.data.dec) )
        # Add tneed 
        kwargs= dict(camera= self.camera,
                     filter_arr= self.data.filter,
                     fwhm_pix= self.data.fwhm,
                     transp= self.data.transp,
                     airmass= self.data.airmass,
                     ebv= self.data.ebv,
                     msky_ab_per_arcsec2= self.data.skymag)
        self.data.set('tneed', CalculateTneed().calc(**kwargs) )
 
    def apply_cuts(self):
        self.zpt_succeeded()
        #self.remove_duplicate_expnum()

    def zpt_succeeded(self):
        keep= (np.isfinite(self.data.zpt))*\
              (self.data.zpt > 0)*\
              (self.data.fwhm > 0)
        print('zpt_succeeded: keeping %d/%d' % \
               (np.where(keep)[0].size,len(keep)))
        self.data.cut(keep)

    def remove_duplicate_expnum(self):
        # Keep everything unless told otherwise
        keep= np.ones(len(self.data),bool)
        # Loop over exposures
        num_dup_exp=0
        num_dup_ccds=0
        expnums= set(self.data.expnum)
        print('Removing duplicates')
        for cnt,expnum in enumerate(expnums):
            if cnt % 100 == 0: 
                print('%d/%d' % (cnt,len(expnums)))
            exp_inds= self.data.expnum == expnum
            if self.camera == 'decam':
                max_sz= 62
            elif self.camera == 'mosaic':
                max_sz= 4
            # Have duplicates?
            if len(self.data[exp_inds]) > max_sz:
                print('Found duplicate expnum=%s' % expnum)
                num_dup_exp += 1
                for ccdname in set(self.data.ccdname[exp_inds]):
                    inds= np.where( (exp_inds)*\
                                    (self.data.ccdname == ccdname) )[0]
                    # We can delete at least one of these
                    if inds.size > 1:
                        num_dup_ccds += inds.size-1
                        keep[ inds[1:] ] = False
        print('number duplicate exposures=%d, ccds=%d' % (num_dup_exp,num_dup_ccds))
        self.data.cut(keep)


class MatchedAnnotZpt(object):
    def __init__(self,decam=None,decam_ann=None,
                      mosaic=None,mosaic_ann=None,
                      bass=None,bass_ann=None):
        """"decam,decam_ann,...: file name of fits tables"""
        self.decam= decam
        self.decam_ann= decam_ann
        self.mosaic= mosaic
        self.mosaic_ann= mosaic_ann
        self.bass= bass
        self.bass_ann= bass_ann
        self.T= defaultdict(dict)
        self.cameras= []
        for camera in CAMERAS:
            if getattr(self,camera):
                self.cameras.append(camera)
                if os.path.exists(self.savefn(getattr(self,camera))):
                    self.T[camera]['z']= fits_table(self.savefn(getattr(self,camera)))
                    self.T[camera]['a']= fits_table(self.savefn(getattr(self,camera+"_ann")))
                else:
                    self.T[camera]['z']= fits_table(getattr(self,camera))
                    self.T[camera]['a']= fits_table(getattr(self,camera+"_ann"))
                    self.match(camera)
                    self.T[camera]['z'].writeto(self.savefn(getattr(self,camera)))
                    self.T[camera]['a'].writeto(self.savefn(getattr(self,camera+"_ann")))

    def __getitem__(self,tup):
        """z_or_a is zpt or annotated"""
        camera,z_or_a= tup
        return self.T[camera][z_or_a]

    def match(self,camera):
        self.clean(camera)
        self.rename_cols(camera)
        self.add_corrected_depth(camera)
        self.same_expid(camera)
        self.drop_duplicates(camera)
        self.row_matched(camera)

    def same_expid(self,camera):
        ind_a= (pd.Series(np.char.strip(self[camera,'a'].expid))
                .isin(np.char.strip(self[camera,'z'].expid))
                .values)
        self[camera,'a'].cut(ind_a)
        print('After same_expid cuts %s,annot: remaining %d/%d' % (camera,len(self[camera,'a']),len(ind_a)))
        ind_z= (pd.Series(np.char.strip(self[camera,'z'].expid))
                .isin(np.char.strip(self[camera,'a'].expid))
                .values)
        self[camera,'z'].cut(ind_z)
        print('After same_expid cuts %s,zpt: remaining %d/%d' % (camera,len(self[camera,'z']),len(ind_z)))

    def drop_duplicates(self,camera):
        for a_or_z in 'az':
            if len(set(self[camera,a_or_z].expid)) != len(self[camera,a_or_z]):
                df= pd.DataFrame(dict(expid=self[camera,a_or_z].expid))
                df.drop_duplicates("expid",inplace=True)
                n_dup=len(self[camera,a_or_z]) - len(df['expid'])
                print('%s,%s dropping %d duplicates' % (camera,a_or_z,n_dup)) 
                self.T[camera][a_or_z]= self[camera,a_or_z][df.index.values]

    def row_matched(self,camera):
        """The a and z tables have the same size and expids but not row matched"""
        for a_or_z in 'az':
            i= np.argsort(self[camera,a_or_z].expid)
            self.T[camera][a_or_z]= self[camera,a_or_z][i]
        assert(np.all(self.T[camera,'a'] == self.T[camera,'z']))
        

    def clean(self,camera):
        # Annot
        keep= ((self[camera,'a'].psfnorm_mean > 0) &
               (self[camera,'a'].photometric) &
               (self[camera,'a'].galnorm_mean > 0) &
               (np.isfinite(self[camera,'a'].psfnorm_mean)) &
               (np.isfinite(self[camera,'a'].galnorm_mean)))
        if camera in ['mosaic','bass']:
            keep= (keep)&(self[camera,'a'].bitmask == 0) 
        self[camera,'a'].cut(keep)
        print('After cuts for %s,annot: remaining %d/%d' % (camera,len(self[camera,'a']),len(keep)))
        # Zpt
        #['fwhm','zpt','skymag']:
        for key in ['fwhm','zpt','skymag']:
            #keep= np.isfinite(self[camera,'z'].get(key))
            keep= self[camera,'z'].get(key) > 0.
            if not np.all(keep):
                self[camera,'z'].cut(keep)
                #print('%s zpt, %s has NANs, remaining %d/%d' % (camera,key,len(self[camera,'z']),len(keep)))
                print('%s %s has < 0 values, cutting to %d/%d' % (camera,key,len(self[camera,'z']),len(keep)))
        #for key in self[camera,'z'].get_columns():
        #    if not np.all(np.isfinite(self[camera,'z'].get(key))):
        #        print('%s zpt, %s has nans')
            

    def rename_cols(self,camera):
        for key in ['ra','dec']:
            # annotated ra,dec center is "ra,dec_center" NOT ra,dec
            self[camera,'a'].set(key,self[camera,'a'].get('%s_center' % key))
        if camera == 'mosaic':
            self[camera,'a'].set('expid',np.char.upper(self[camera,'a'].expid))
            

    def savefn(self,fits_fn):
        return (fits_fn
                .replace(".gz","")
                .replace(".fits","_matched.fits"))

    def add_corrected_depth(self,camera):
        D= Depth(camera,
                 self[camera,'z'].skyrms,self[camera,'z'].gain,
                 self[camera,'z'].fwhm,self[camera,'z'].zpt)
        # Neff corrected
        self[camera,'z'].set('psfdepth', D.get_depth_legacy_zpts('psf',func_neff=NeffFormulas().neff_empirical))
        self[camera,'z'].set('galdepth', D.get_depth_legacy_zpts('gal',func_neff=NeffFormulas().neff_empirical))
        # Neff default formula
        self[camera,'z'].set('psfdepth_uncorr', D.get_depth_legacy_zpts('psf',func_neff=None))
        self[camera,'z'].set('galdepth_uncorr', D.get_depth_legacy_zpts('gal',func_neff=None))
        for name in ['psf','gal']:
            self[camera,'a'].set('%sdepth_uncorr' % name, self[camera,'a'].get('%sdepth' % name))
        # Can induce Nans
        keep= ((np.isfinite(self[camera,'z'].galdepth)) &
               (np.isfinite(self[camera,'z'].psfdepth)))
        self[camera,'z'].cut(keep)
        print('After add depth, %s non-Nan remaining %d/%d' % (camera,len(self[camera,'z']),len(keep)))



class NeffFormulas(object):
    def neff_15(self,seeing,rhalf=0.45,pix=0.262):
	    return 4*np.pi*seeing**2 + 8.91*rhalf**2 + pix**2/12

    def neff_empirical(self,seeing,camera,psf_or_gal):
        """Correction for Neff estimator"""
        if psf_or_gal == 'psf':
            rhalf=0
            if camera == 'decam':
                slope= 1.29
                yint= 6.8
            elif camera == 'mosaic':
                slope= 1.33
                yint= 2.6
        elif psf_or_gal == 'gal':
            rhalf=0.45
            if camera == 'decam':
                slope= 1.31
                yint= 42.6
            elif camera == 'mosaic':
                slope= 1.36
                yint= 37.6
        return slope * self.neff_15(seeing,rhalf=rhalf) + yint

class LeastSquares(object):
    def __init__(self,x,y,yerr):
        self.sum_inv_yerr2= np.sum(1/yerr**2)
        self.sum_y_inv_yerr2= np.sum(y/yerr**2)
        self.sum_x_inv_yerr2= np.sum(x/yerr**2)
        self.sum_xy_inv_yerr2= np.sum(x*y/yerr**2)
        self.sum_x2_inv_yerr2= np.sum(x**2/yerr**2)
        self.delta= self.sum_inv_yerr2 * self.sum_x2_inv_yerr2 - self.sum_x_inv_yerr2**2
        
    def estim(self):
        return dict(slope=self.slope(),
                    inter=self.inter(),
                    s_err=self.err_slope(),
                    i_err=self.err_inter())
    
    def slope(self):
        return 1/self.delta * (self.sum_inv_yerr2*self.sum_xy_inv_yerr2 - \
                               self.sum_x_inv_yerr2 * self.sum_y_inv_yerr2)
    
    def inter(self):
        return 1/self.delta * (self.sum_x2_inv_yerr2*self.sum_y_inv_yerr2 - \
                               self.sum_x_inv_yerr2 * self.sum_xy_inv_yerr2)
    
    def err_slope(self):
        return 1/self.delta * self.sum_inv_yerr2
    
    def err_inter(self):
        return 1/self.delta * self.sum_x2_inv_yerr2

def getrms(x):
    return np.sqrt( np.mean( np.power(x,2) ) )

def firstcap(text):
    return "%s%s" % (text[0].upper(),text[1:].lower())

def get_q25(x):
    return np.percentile(x,q=25)

def get_q50(x):
    return np.percentile(x,q=50)

def get_q75(x):
    return np.percentile(x,q=75)

class NeffPlots(object):
    def __init__(self):
        self.cam2color=dict(decam='g',mosaic='m',bass='r')
        self.which2shape=dict(psf='o',gal='s')
        self.rhalf=dict(psf=0., gal=0.45)
        self.pix=dict(decam=0.262,mosaic=0.26,bass=0.455)
        self.params= defaultdict(dict)

    def neff(self,M,which='residual'):
        """M:  MatchedAnnotZpt() object"""
        assert(which in ['residual_v_truth','truth_v_model']) 
        if which == 'residual_v_truth':
            self.neff_residual_v_truth(M)
        elif which == 'truth_v_model':
            self.neff_fit_and_plot(M)  

    def neff_residual_v_truth(self,M):
        # best fit computed elsewhere
        if not self.params:
            self.neff_fit_and_plot(M)
        # PSF Neff vs. 1/norm^2
        #offsets=np.arange(0,500*factor/2,100*factor/2)
        xlim=dict(decam=dict(psf=(20,250),gal=(50,300)),
                  mosaic=dict(psf=(20,250),gal=(50,300)))
        ylim=dict(decam=dict(psf=(-30,30),gal=(-30,30)),
                  mosaic=dict(psf=(-30,30),gal=(-30,30)))
        nbins= (40,40)
        fig, ax = plt.subplots(2,2,figsize=(8, 6))
        plt.subplots_adjust(hspace=0,wspace=0.1)
        i=-1
        for row,camera in enumerate(M.cameras):
            for col,which in enumerate(['psf','gal']):
                i+=1
                color= self.cam2color[camera]
                #offset= offsets[i]
                seeing= M[camera,'z'].fwhm/2.35
                y=1./M[camera,'a'].get('%snorm_mean' % which)**2
                model=NeffFormulas().neff_15(seeing,rhalf=self.rhalf[which],pix=self.pix[camera])
                model=self.params[camera][which]['slope']*model + self.params[camera][which]['inter']
                myhist2D(ax[row,col],y,y-model,
                         xlim=xlim[camera][which],ylim=ylim[camera][which],nbins=nbins)
                #plt.scatter(y,(y-model)*factor+offset, c=color,edgecolors='none',marker=self.which2shape[which],s=10,rasterized=True,alpha=alpha,label='%s (%s)' % (camera,which))
                # rms,q7525
                #rms= getrms(y-model)
                #x2=np.linspace(0,700) 
                ax[row,col].axhline(0,c='r',ls='dashed',lw=1)
                mytext(ax[row,col],0.05,0.95,"%s, %s" % (camera,which), 
                       ha='left',va='center',fontsize=12)
                mytext(ax[row,col],0.05,0.88,r"($N_{eff} \approx %.2f \hat{N}_{eff} + %.1f$)" % \
                        (self.params[camera][which]['slope'],self.params[camera][which]['inter']), 
                       ha='left',va='center',fontsize=8)
        # Label
        #plt.legend(loc='upper right',ncol=1,fontsize=10,markerscale=3)
        #plt.xlim(0,900);plt.ylim(0,700)
        for col in range(2):
            xlab=ax[1,col].set_xlabel('Neff (Truth)')
            ax[0,col].xaxis.set_ticklabels([])
        for row in range(2):
            ylab=ax[row,0].set_ylabel('Residual (truth-model)') #'1/{psf,gal}norm_mean^2')
            ax[row,1].yaxis.set_ticklabels([])
        savefn= 'neff_residual.png'
        plt.savefig(savefn,bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
        print('Wrote %s' % savefn)
        plt.close()


    def neff_fit_and_plot(self,M):
        # PSF Neff vs. 1/norm^2
        offsets=[0,100,200,300,400,500]
        i=-1
        for camera in M.cameras:
            for which in ['psf','gal']:
                i+=1
                color= self.cam2color[camera]
                offset= offsets[i]
                seeing= M[camera,'z'].fwhm/2.35
                x=NeffFormulas().neff_15(seeing,rhalf=self.rhalf[which],pix=self.pix[camera])
                y=1./M[camera,'a'].get('%snorm_mean' % which)**2
                plt.scatter(x+offset,y, c=color,edgecolors='none',marker=self.which2shape[which],s=10,rasterized=True,alpha=0.75)
                self.params[camera][which]= LeastSquares(x,y,
                                                         np.ones(len(y))).estim()
                # rms,q7525
                diff= y - self.params[camera][which]['inter']-self.params[camera][which]['slope']*x
                rms= getrms(diff)
                x2=np.linspace(0,700) 
                lab= '%s,%s: y=%.2fx+%.1f; RMS=%.1f' % (camera,which,self.params[camera][which]['slope'],self.params[camera][which]['inter'],rms)
                plt.plot(x2+offset,self.params[camera][which]['slope']*x2+self.params[camera][which]['inter'],color,ls='dashed',lw=2,label=lab)
        # Label
        leg=plt.legend(loc=(0.,1.02),ncol=2,fontsize=10)
        plt.xlim(0,900);plt.ylim(0,700)
        xlab=plt.xlabel('Neff (Estimator) + offset')
        ylab=plt.ylabel('Neff (Truth)') #'1/{psf,gal}norm_mean^2')
        savefn= 'neff_plot.png'
        plt.savefig(savefn,bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight',dpi=150)
        print('Wrote %s' % savefn)
        plt.close()

    def depth_residual(self,M,which='gal'):
        FS=14
        eFS=FS+5
        tickFS=FS

        fig,axes= plt.subplots(2,2,figsize=(8,4))
        ax= axes.flatten()
        plt.subplots_adjust(hspace=0.2,wspace=0.1)
       
        if which == 'gal': 
            xlim=dict(decam=dict(g=(22.,24.5),r=(22.,24.),z=(20.,23.5)),
                      mosaic=dict(z=(20.,23.5)))
            nbins=dict(decam=dict(g=(20,15),r=(20,15),z=(20,15)),
                       mosaic=dict(z=(40,30)))
            ylim=(-0.2,0.2)
        else:
            xlim=dict(decam=dict(g=(23.,25.),r=(22.,24.5),z=(20,24)),
                      mosaic=dict(z=(20,24)))
            nbins=dict(decam=dict(g=(20,15),r=(20,15),z=(20,15)),
                       mosaic=dict(z=(40,30)))
            ylim=(-0.25,0.25)
        #gridsize=dict(g=(20,20),r=(25,8),z=30)
        #hb= defaultdict(dict)

        dreq= DepthRequirements() 

        for i,camera,band in [(0,'decam','g'),(1,'decam','r'),
                              (2,'decam','z'),(3,'mosaic','z')]:
            color= self.cam2color[camera]
            key= which+'depth'
            keep= M[camera,'a'].filter == band
            if len(M[camera,'a'][keep]) > 0:
                y= M[camera,'a'].get(key)[keep]
                model= M[camera,'z'].get(key)[keep]
                myhist2D(ax[i],y,y-model,
                         xlim=xlim[camera][band],ylim=ylim,nbins=nbins[camera][band])
                #hb[i] = ax[i].hexbin(y,y-model, gridsize=gridsize[band],
                #                     cmap='gray_r',bins='log') #vmin=0,vmax=vmax[camera],
                ax[i].axhline(0,c='r',ls='--')
                one_pass= dreq.get_single_pass_depth(band,which,camera)
                ax[i].axvline(one_pass,c='b',ls='--')
                mytext(ax[i],0.4,0.1,"%s, %s (%.2f)" % (camera,band,one_pass), 
                       ha='left',va='center',fontsize=FS)

                # percentile lines
                #new= pd.DataFrame(dict(y=y,model=model))
                #new['diff']= new['y'] - new['model']
                #new['bins']= pd.cut(new['y'],bins=15)
                #a= new.groupby('bins').agg([get_q25,get_q50,get_q75])
                #binc= a.index.categories.mid
                #ax[row,col].plot(binc,a['diff']['get_q25'],'b-')
                #ax[row,col].plot(binc,a['diff']['get_q50'],'b-')
                #ax[row,col].plot(binc,a['diff']['get_q75'],'b-')
            
        # Label
        for i in [2,3]:
            xlab=ax[i].set_xlabel('%s (truth)' % firstcap(which+'depth'),fontsize=FS)
        for i in [0,2]:
            ylab=ax[i].set_ylabel('Residual\n(truth - model)',fontsize=FS)
        for i in [1,3]:
            ax[i].yaxis.set_ticklabels([])
        savefn='%sdepth_residual.png' % which
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close()
        print("wrote %s" % savefn)

class histsAtDiffMJDs(object):
    def plot_hist_at_diff_mjd(self,Z,M,camera,key):
        """
        Args:
            Z: ZeropointHistogram() object
            M: MatchedAnnotZpt() object
        """
        assert(key in ['zpt','transp'])
        #self.convert_idl2legacy(M,camera,key)
       
        import seaborn as sns
        #mjdBoundary= dict(mosaic=M['decam','a'].mjd_obs.max(),
        #                  decam=M['mosaic','a'].mjd_obs.max())
        #isNew= getattr(Z,camera).mjd_obs > mjdBoundary[camera]
        #isNew= dict(M= M[camera,'z'].mjd_obs > mjdBoundary[camera],
        #            Z= getattr(Z,camera).mjd_obs > mjdBoundary[camera])
        xlim=dict(mosaic=dict(transp=(0.4,1.),
                              zpt=(25.8,26.6)),
                  decam=dict(transp=(0.5,1.5),
                             zpt=(26.1,27.2)))
        mjdBinSize=dict(decam=300,mosaic=100)
        #jitter=dict(decam=dict(zpt=0.,transp=0.),
        #            mosaic=dict(zpt=0.01,transp=0.008))
        label=dict(mosaic='DR4',decam='DR3')

        fig,ax=plt.subplots(1,2,figsize=(10,4))

        kwargs= dict(kde=True,hist=False,
                     hist_kws={"histtype": "step",'lw':2})

        #if key == 'transp':
        #    keep= M[camera,'a'].get('ccd'+key+'_legunits') < 2
        #elif key == 'zpt':
        #    keep= ((M[camera,'a'].get('ccd'+key+'_legunits') < 29) & 
        #           (M[camera,'a'].get('ccd'+key+'_legunits') > 20))

        #sns.distplot(M[camera,'a'].get('ccd'+key+'_legunits')[keep],
        #             ax=ax[0],label=label[camera]+' Zpts',**kwargs) 
        #sns.distplot(M[camera,'z'].get(key)[keep]+jitter[camera][key],ax=ax[0],label='LegacyZpts + %.1g (epoch < %s)' % (jitter[camera][key],label[camera]),**kwargs)
        mjdBins= np.arange(getattr(Z,camera).mjd_obs.min(),
                           getattr(Z,camera).mjd_obs.max(),mjdBinSize[camera])
        mjdPrev= mjdBins[0]
        for mjdBoundary in mjdBins[1:]:
            keep= ((getattr(Z,camera).mjd_obs >= mjdPrev) &
                   (getattr(Z,camera).mjd_obs < mjdBoundary))
            sns.distplot(getattr(Z,camera).get(key)[keep],ax=ax[0],**kwargs)
            sns.distplot(getattr(Z,camera).mjd_obs[keep],ax=ax[1],**kwargs)
            mjdPrev=mjdBoundary
        #sns.distplot(M[camera,'z'].get(key)[~isNew],ax=ax[0],label='LegacyZpts',**kwargs)
        ax[0].legend()
        ax[0].set_xlim(xlim[camera][key])
        ax[0].set_xlabel(key)
        ylab=ax[0].set_ylabel('PDF')

        #sns.distplot(M[camera,'a'].mjd_obs[keep],ax=ax[1],label=label[camera]+' Zpts',**kwargs) 
        #sns.distplot(M[camera,'z'].mjd_obs[keep]+jitter[camera][key],ax=ax[1],label='LegacyZpts + %.1g (epoch < %s)' % (jitter[camera][key],label[camera]),**kwargs)
        #sns.distplot(getattr(Z,camera).mjd_obs[isNew],ax=ax[1],label='LegacyZpts (epoch > %s)' % label[camera],**kwargs)
        #ax[1].legend()
        xlab=ax[1].set_xlabel('MJD')
        for i in range(2):
            mytext(ax[i],0.7,0.95,camera, 
                   ha='left',va='center',fontsize=12)

        savefn='hist_atdiffmjd_%s_%s.png' % (camera,key)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab],bbox_inches='tight',dpi=150)
        print("wrote %s" % savefn)

    def plot_hist_at_zpt_boundary(self,Z,kde=False):
        """
        Args:
            Z: ZeropointHistogram() object
        """
        if kde:
            import seaborn as sns
        else:
            from matplotlib.colors import ListedColormap,BoundaryNorm
        camera='decam'
        key='zpt'
        #self.convert_idl2legacy(M,camera,key)
       
        #mjdBoundary= dict(mosaic=M['decam','a'].mjd_obs.max(),
        #                  decam=M['mosaic','a'].mjd_obs.max())
        #isNew= getattr(Z,camera).mjd_obs > mjdBoundary[camera]
        #isNew= dict(M= M[camera,'z'].mjd_obs > mjdBoundary[camera],
        #            Z= getattr(Z,camera).mjd_obs > mjdBoundary[camera])
        xlim=dict(g=(26.2,27.),r=(26.4,27.1),z=(26.,26.7))
        zptThresh=dict(g=26.7,r=26.875,z=26.475)
        #mjdBinSize=dict(decam=300,mosaic=100)
        #jitter=dict(decam=dict(zpt=0.,transp=0.),
        #            mosaic=dict(zpt=0.01,transp=0.008))
        
        label=dict(mosaic='DR4',decam='DR3')

        fig,axes=plt.subplots(3,2,figsize=(6,8))
        plt.subplots_adjust(wspace=0.4,hspace=0.2)
        ax=axes.flatten()

        kwargs= dict(kde=True,hist=False,
                     hist_kws={"histtype": "step",'lw':2})

        #if key == 'transp':
        #    keep= M[camera,'a'].get('ccd'+key+'_legunits') < 2
        #elif key == 'zpt':
        #    keep= ((M[camera,'a'].get('ccd'+key+'_legunits') < 29) & 
        #           (M[camera,'a'].get('ccd'+key+'_legunits') > 20))

        #sns.distplot(M[camera,'a'].get('ccd'+key+'_legunits')[keep],
        #             ax=ax[0],label=label[camera]+' Zpts',**kwargs) 
        #sns.distplot(M[camera,'z'].get(key)[keep]+jitter[camera][key],ax=ax[0],label='LegacyZpts + %.1g (epoch < %s)' % (jitter[camera][key],label[camera]),**kwargs)
        for i,band in [(0,'g'),(2,'r'),(4,'z')]:
            isBand= getattr(Z,camera).filter == band
            isBefore= getattr(Z,camera).zpt < zptThresh[band]
            if kde:
                sns.distplot(getattr(Z,camera).get(key)[isBand],ax=ax[i],**kwargs)
            else:
                # Create a colormap for red, green and blue and a norm to color
                # f' < -0.5 red, f' > 0.5 blue, and the rest green
                bins= np.linspace(xlim[band][0],xlim[band][1],num=40)
                
                hist, _ = np.histogram(getattr(Z,camera).get(key)[isBefore & isBand], 
                                       bins=bins, density=True)
                hist *= len(getattr(Z,camera)[isBefore & isBand])/len(getattr(Z,camera)[isBand])
                ax[i].step(bins[:-1], hist, where='post',c='g')
                hist, _ = np.histogram(getattr(Z,camera).get(key)[~isBefore & isBand], 
                                       bins=bins, density=True)
                hist *= len(getattr(Z,camera)[~isBefore & isBand])/len(getattr(Z,camera)[isBand])
                ax[i].step(bins[:-1], hist, where='post',c='b')

                #myhist_step(ax[i],getattr(Z,camera).get(key)[isBefore & isBand], bins=bins,
                #            normed=True,ls='solid',color='g') #color=band2color(band),ls='solid',
                #myhist_step(ax[i],getattr(Z,camera).get(key)[~isBefore & isBand], bins=bins,
                #            normed=True,ls='solid',color='b') #color=band2color(band),ls='solid',
            ax[i].axvline(zptThresh[band],c='k',ls='--')
            mytext(ax[i],zptThresh[band],ax[i].get_ylim()[1]*0.95,'%.2f' % zptThresh[band], 
                   ha='left',va='center',fontsize=12,dataCoords=True) 
            if kde:
                sns.distplot(getattr(Z,camera).mjd_obs[isBefore],ax=ax[i+1],
                             label='zpt < %.2f' % zptThresh[band],**kwargs)
                sns.distplot(getattr(Z,camera).mjd_obs[~isBefore],ax=ax[i+1],
                             label='otherwise',**kwargs)
            else:
                bins= np.linspace(getattr(Z,camera).mjd_obs.min(),
                                  getattr(Z,camera).mjd_obs.max(),40)
                
                hist, _ = np.histogram(getattr(Z,camera).mjd_obs[isBefore & isBand], 
                                       bins=bins, density=True)
                hist *= len(getattr(Z,camera)[isBefore & isBand])/len(getattr(Z,camera)[isBand])
                ax[i+1].step(bins[:-1], hist, where='post',c='g',
                           label='zpt < %.2f' % zptThresh[band])
                hist, _ = np.histogram(getattr(Z,camera).mjd_obs[~isBefore & isBand], 
                                       bins=bins, density=True)
                hist *= len(getattr(Z,camera)[~isBefore & isBand])/len(getattr(Z,camera)[isBand])
                ax[i+1].step(bins[:-1], hist, where='post',c='b',
                           label='otherwise')
                
                #myhist_step(ax[i+1],getattr(Z,camera).mjd_obs[isBefore & isBand], bins=bins,
                #            normed=True,ls='solid',color='g',label='zpt < %.2f' % zptThresh[band])
                #myhist_step(ax[i+1],getattr(Z,camera).mjd_obs[~isBefore & isBand], bins=bins,
                #            normed=True,ls='solid',color='b',label='otherwise')

            ax[i+1].legend(loc='upper right',fontsize=8)
            mytext(ax[i],0.05,0.92,band, 
                   ha='left',va='center',fontsize=12)
            mytext(ax[i+1],0.05,0.92,band, 
                   ha='left',va='center',fontsize=12)
            ax[i].set_xlim(xlim[band])
        #sns.distplot(M[camera,'z'].get(key)[~isNew],ax=ax[0],label='LegacyZpts',**kwargs)
        ax[4].set_xlabel(key)
        for i in [0,2,4]:
            ylab=ax[i].set_ylabel('PDF')
        xlab=ax[4].set_xlabel('Zeropoint')
        xlab=ax[5].set_xlabel('MJD')
        #sns.distplot(M[camera,'a'].mjd_obs[keep],ax=ax[1],label=label[camera]+' Zpts',**kwargs) 
        #sns.distplot(M[camera,'z'].mjd_obs[keep]+jitter[camera][key],ax=ax[1],label='LegacyZpts + %.1g (epoch < %s)' % (jitter[camera][key],label[camera]),**kwargs)
        #sns.distplot(getattr(Z,camera).mjd_obs[isNew],ax=ax[1],label='LegacyZpts (epoch > %s)' % label[camera],**kwargs)
        #ax[1].legend()
        savefn='hist_at_zpt_boundary_%s_%s.png' % (camera,key)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab],bbox_inches='tight',dpi=150)
        print("wrote %s" % savefn)


    
    def convert_idl2legacy(self,M,camera,key):
        """dr is dr3 or dr4 survey-ccds fit table"""
        assert(key in ['zpt','transp'])
        if camera == 'mosaic':
            val= M[camera,'a'].get('ccd'+key)
        elif camera == 'decam':
            val= M[camera,'a'].get('ccd'+key)
            if key == 'zpt':
                val= M[camera,'a'].get('ccd'+key) + 2.5*np.log10(M[camera,'a'].arawgain)
            #elif key == 'transp':
            #    val= M[camera,'a'].get('ccd'+key) / (2.5*np.log10(M[camera,'a'].arawgain))
        M.T[camera]['a'].set('ccd'+key+'_legunits',val)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                            description='Generate a legacypipe-compatible CCDs file \
                                        from a set of reduced imaging.')
    parser.add_argument('--figs',type=str,choices=["1-8,11","1a-8a","5a","9-10"],help='legacyzpts "zpt" ccd file',required=False)
    parser.add_argument('--plot_all',action="store_true",default=False,help='legacyzpts "zpt" ccd file',required=False)
    parser.add_argument('--decam',type=str,default=None,help='legacyzpts "zpt" ccd file',required=False)
    parser.add_argument('--decam_ann',type=str,default=None,help='annotated ccd file',required=False)
    parser.add_argument('--mosaic',type=str,default=None,required=False)
    parser.add_argument('--mosaic_ann',type=str,default=None,required=False)
    parser.add_argument('--bass',type=str,default=None,required=False)
    parser.add_argument('--bass_ann',type=str,default=None,required=False)
    args = parser.parse_args()

    kwargs= vars(args)
    figs= kwargs['figs'] 
    plot_all= kwargs['plot_all'] 
    for key in ['figs','plot_all']:
        del kwargs[key]
   
    data= {}
    if figs in ["1-8,11","1a-8a"] or plot_all:
        data['Z']= ZeropointHistograms(decam=args.decam,
                                       mosaic=args.mosaic,
                                       bass=args.bass)
        if figs == "1-8,11" or plot_all:
            # histograms of ccd stats
            data['Z'].plot()
        elif figs == "1a-8a" or plot_all:
            # how zpt and raoff signif. change with time
            data['Z'].plot_v_mjd('decam')
    if figs in ["9-10"] or plot_all:
        # Neff fits and Galdepth predictions vs truth
        data['M']= MatchedAnnotZpt(**kwargs)

        if figs == "9-10" or plot_all:
            NeffPlots().neff(data['M'],'truth_v_model')
            NeffPlots().neff(data['M'],'residual_v_truth')
            for which in ['gal','psf']:
                NeffPlots().depth_residual(data['M'],which)
    if figs == "5a" or plot_all:
        # Bimodality in zpt due to newer obs sys diff from zp0
        if not ('Z' in data.keys()):
            data['Z']= ZeropointHistograms(decam=args.decam,
                                           mosaic=args.mosaic,
                                           bass=args.bass)
        if not ('M' in data.keys()):
            data['M']= MatchedAnnotZpt(**kwargs)

        histsAtDiffMJDs().plot_hist_at_zpt_boundary(data['Z'])
        raise ValueError("stopping here")
        for key in ['zpt','transp']:
            for camera in ['decam']: #M.cameras:
                data['Z'].plot_v_mjd(camera)
                histsAtDiffMJDs().plot_hist_at_diff_mjd(data['Z'],data['M'],camera,key)

