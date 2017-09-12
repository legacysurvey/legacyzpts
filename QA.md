# QA for DECaLS, MzLS, BASS

# Run 4 CCDs of DECaLS, MzLS, or BASS 
```
python py/legacyzpts/legacy_zeropoints.py --camera 90prime --image /project/projectdirs/cosmo/staging/bok/BOK_CP/CP20160603/ksb_160604_102744_ooi_r_v1.fits.fz --outdir test --debug
```

# Make QA plots
```
/project/projectdirs/cosmo/staging/decam/DECam_CP/CP20170326/c4d_170326_233934_oki_z_v1.fits.fz
/project/projectdirs/cosmo/staging/decam/DECam_CP/CP20170326/c4d_170327_042837_oki_g_v1.fits.fz
/project/projectdirs/cosmo/staging/decam/DECam_CP/CP20170326/c4d_170327_042342_oki_r_v1.fits.fz
```

```
export datadir=/global/cscratch1/sd/kaylanb/dr5_zpts/decam
export arjundir=/global/cscratch1/sd/kaylanb/arjundey_Test/AD_exact_skymed
```

```
python legacy_zeropoints.py 
```

# Run The Code 

# Databases

# License

legacyzpts is free software licensed under a 3-clause BSD-style license. For details see
the ``LICENSE.rst`` file.
