import numpy as np
from collections import defaultdict


class EmptyClass(object): 
    pass


def getrms(x):
    return np.sqrt( np.mean( np.power(x,2) ) ) 

def band2color(band):
    d=dict(g='g',r='r',z='m')
    return d[band]

def col2plotname(key):
    d= dict(airmass= 'Airmass',
            fwhm= 'FWHM (pixels)',
            seeing= 'FWHM (arcsec)',
            gain= 'Gain (e/ADU)',
            skymag= 'Sky Brightness (AB mag/arcsec^2)',
            skyrms= 'Sky RMS (e/sec/pixel)',
            transp= 'Atmospheric Transparency',
            zpt= 'Zeropoint (e/s)',
            err_on_radecoff= 'Astrometric Offset Error (Std Err of Median, arcsec)',
            err_on_phoff= r'Zeropoint Error (Std Err of Median), mag)',
            psfdepth= r'Point-source Single Pass Depth (5$\sigma$)',
            galdepth= r'Galaxy Single Pass Depth (5$\sigma$)')
    d['psfdepth_extcorr']= r'Point-source Depth (5$\sigma$, ext. corrected)'
    d['galdepth_extcorr']= r'Galaxy Depth (5$\sigma$, ext. corrected)'
    return d.get(key,key)



def get_fiducial(camera=None):
    assert(camera in ['decam','mosaic'])
    obj= EmptyClass()
    obj.camera = camera 
    obj.desi_mags_1pass= defaultdict(lambda: defaultdict(dict))
    obj.desi_mags= defaultdict(lambda: defaultdict(dict))
    obj.desi_mags['gal']['g']= 24.0
    obj.desi_mags['gal']['r']= 23.4
    obj.desi_mags['gal']['z']= 22.5
    obj.desi_mags['psf']['g']= 24.7
    obj.desi_mags['psf']['r']= 23.9
    obj.desi_mags['psf']['z']= 23.0
    if camera == 'decam':
        obj.bands= ['g','r','z']
        obj.pixscale= 0.262
        obj.gain= 4.34
        obj.zp0= dict(g=26.610, r=26.818, z=26.484) # e/s
        obj.sky0= dict(g=22.04, r=20.91, z=18.46)
        obj.tmin= dict(g=56,r=40,z=80)
        obj.tmax= dict(g=175,r=125,z=250)
        obj.t0= dict(g=70,r=50,z=100)
        obj.Aco= dict(g=3.214,r=2.165,z=1.562)
        obj.Kco= dict(g=0.17,r=0.1,z=0.06)
        obj.bothuman_list= ['bot','human']
        # required depth after 1 pass
        for psf_or_gal in ['psf','gal']:
            for band in obj.bands:
                obj.desi_mags_1pass[psf_or_gal][band]= obj.desi_mags[psf_or_gal][band] \
                                                        - 2.5*np.log10(2**0.5)
    elif camera == 'mosaic':
        obj.bands=['z']
        obj.pixscale= 0.260
        obj.gain= 1.8
        obj.zp0= dict(z=26.552) # e/s
        obj.sky0= dict(z=18.46)
        obj.tmin= dict(z=80)
        obj.tmax= dict(z=250)
        obj.t0= dict(z=100)
        obj.Aco= dict(z=1.562)
        obj.Kco= dict(z=0.06)
        obj.bothuman_list= ['bot']
        # required depth after 1 pass
        for psf_or_gal in ['psf','gal']:
            for band in obj.bands:
                obj.desi_mags_1pass[psf_or_gal][band]= obj.desi_mags[psf_or_gal][band] \
                                                        - 2.5*np.log10(3**0.5)
    obj.fwhm0= 1.3 #arcsec
    # Build all other bot_avg,human_avg,fixed lists
    obj.bothuman_avg_list= ['%s_avg' % b for b in obj.bothuman_list]
    obj.fixed_list= ['fixed_%s' % b for b in obj.bothuman_list]
    obj.fixed_avg_list= ['fixed_%s' % b for b in obj.bothuman_avg_list]
    # [bot,human]+[fixed_bot,fixed_human] --> [bot,fixed_bot]
    obj.botfixed_list=       [b for b in obj.bothuman_list + obj.fixed_list
                                if 'bot' in b]
    obj.humanfixed_list=     [b for b in obj.bothuman_list + obj.fixed_list
                                if 'human' in b]
    obj.botfixed_avg_list=   [b for b in obj.bothuman_avg_list + obj.fixed_avg_list
                                if 'bot' in b]
    obj.humanfixed_avg_list= [b for b in obj.bothuman_avg_list + obj.fixed_avg_list
                                if 'human' in b]
    return obj 


