# What Legacyzpts Creates
Four FITS data tables with the same columns and units for all three cameras.

### (1) <image name>-zpt.fits
Most important:

| Name | Type | Units | Description
| ------ | ------ | ------ | ------ |
| airmass | float32 | ---- | airmass from CP header
| camera | str | ---- | decam, mosaic, or 90prime
| ccdname | str | ---- | ccd[1-4] (mosaic,90prime) or [NS][0-9]{2} (decam)
| date_obs | str | ---- | yyyy-mm-dd
| dec | float64 | deg | WCS solution center of the CCD 
| decoff | float32 | deg | Median of 3 sigma-clipped ("Gaia Dec" - dec)
| decrms | float32 | deg | RMS of 3 sigma-clipped ("Gaia Dec" - dec)
| decstddev | float32 | deg | Std Dev of 3 sigma-clipped ("Gaia Dec" - dec)
| expnum | int32 | ---- | primary key
| err_message | str | ---- | empty string if CCD was processed without issue
| exptime | float32 | sec | exposure time
| filter | str | ---- | g, r, or z
| fwhm | float32 | pixel | measured 
| fwhm_cp | float32 | pixel | taken from CP header
| gain | float32 | e-/ADU | (Decam) average of GAINA, GAINB, (Mosaic) GAIN, (90prime) 1.4
| ha | str | N/A | hh:mm:ss.ss
| height | int16 | pixel | height of CCD
| image_filename | str | ---- | path to image on NERSC machines, relative to /project/projectdirs/cosmo/staging
| mjd_obs | float64 | ---- | Modified Julian Date
| nmatch_astrom | int16 |  ---- | number of good detected sources that have good Gaia matches within 1''
| nmatch_photom | int16 | ---- | number of good detected sources that have good PS1 matches within 1''
| phoff | float32 | AB mag | Per CCD Median of 2.5 sigma-clipped (PS1 mag - Our mag)
| phrms | float32 | AB mag | Per CCD Std Dev of 2.5 sigma-clipped (PS1 mag - Our mag)
| pixscale | float32 | as/pixel | fixed at 0.262 (DECam, Mosaic) and 0.470 (90Prime)
| ra | float64 | deg | WCS solution center of the CCD 
| raoff | float32 | deg | Median of 3 sigma-clipped ("Gaia Ra" - ra) * cos(dec)
| rarms | float32 | deg | RMS of 3 sigma-clipped ("Gaia Ra" - ra) * cos(dec)
| rastddev | float32 | deg | Std Dev of 3 sigma-clipped("Gaia Ra" - ra) * cos(dec)
| skycounts | float32 | e-/pixel/sec | Median of 3 sigma-clipped 1000x1000 central pixels of CCD 
| skymag | float32 | AB mag/as^2 | skycounts converted to AB mag/as^2
| skyrms | float32 | e-/pixel/sec | Std Dev of 3 sigma-clipped 1000x1000 central pixels of CCD 
| transp | float32 | ---- | Relative atmospheric transparency
| width | int16 | pixel | width of CCD
| zpt | float32 | e-/sec | phoff + nominal zeropoint for camera, filter
| zptavg | float32 | e-/sec | average zpt over all CCDs
 
Less important:

| Name | Type | Units | Description
| ------ | ------ | ------ | ------ |
| ccdnum | int16 | ---- | ccdname with alphabetic characters removed
| cd1_1 | float32 | ---- | Astrometric solution
| cd1_2 | float32 | ---- | Astrometric solution
| cd2_1 | float32 | ---- | Astrometric solution 
| cd2_2 | float32 | ---- | Astrometric solution
| crpix1 | float32 | ---- | Astrometric solution
| crpix2 | float32 | ---- | Astrometric solution
| crval1 | float64 | ---- | Astrometric solution
| crval2 | float64 | ---- | Astrometric solution
| dec_bore | float64 | deg | bore sight of telescope
| expid | str | ---- | expnum + ccdname
| image_hdu | int16 | ---- | hdu number for the ccd in the image FITS file
| object | str | ---- | type of exposure (object vs. flat)
| propid | str | ---- | proposal id for the survey
| ra_bore | float64 | deg | bore sight of telescope
| ut | str | ---- | hh:mm:ss.ss 

### (2) <image name>-legacyzpt.fits
All of these are duplicates of #1 and are the minimal set required by our legacypipe/Tractor pipeline. Note, the names and units are different from above.

| Name | Type | Decam | Mosaic/90Prime | Description
| ------ | ------ | ------ | ------ | ------ |
| camera | str | ---- | ---- | same as above
| ccddecoff | float64 | deg | deg | decoff
| ccdname | str | ---- | ---- | same as above
| ccdnmatch | int16 | ---- | ---- | nmatch_photom
| ccdraoff | float64 | deg | deg | raoff
| ccdzpt | float32 | ADU/sec | e-/sec | zpt
| cd1_1 | float32 | ---- | ---- | same as above
| cd1_2 | float32 | ---- | ---- | same as above
| cd2_1 | float32 | ---- | ---- | same as above
| cd2_2 | float32 | ---- | ---- | same as above
| crpix1 | float32 | ---- | ---- | same as above
| crpix2 | float32 | ---- | ---- | same as above
| crval1 | float64 | ---- | ---- | same as above
| crval2 | float64 | ---- | ---- | same as above
| dec | float64 | deg | deg | same as above
| dec_bore | float64 | deg | deg | same as above
| expnum | int32 | ---- | ---- | same as above
| exptime | float32 | sec | sec | same as above
| filter | str | ---- | ---- | same as above
| fwhm | float32 | pixel | pixel | same as above
| height | int16 | pixel | pixel | same as above
| image_filename | str | ---- | ---- | same as above
| image_hdu | int16 | ---- | ---- | same as above
| mjd_obs | float64 | ---- | ---- | same as above
| object | str | ---- | ---- | same as above
| propid | str | ---- | ---- | same as above
| ra | float64 | deg | deg | same as above
| ra_bore | float64 | deg | deg | same as above
| skyrms | float32 | e-/pixel/sec | e-/pixel/sec | same as above
| width | int16 | pixel | pixel | same as above
| zpt | float32 | ADU/sec | e-/sec | zptavg

### (3) Two stars tables
There are two stars tables: <image name>-star-photom.fits and <image name>-star-astrom.fits. Both have this base set of columns

| Name | Type | Units | Description
| ------ | ------ | ------ | ------ |
| ccdname | str | ---- | same as above
| dec | float64 | deg | declination of detected source
| expnum | int32 | ---- | same as above
| expid | str | ---- | same as above
| exptime | float32 | sec | same as above
| filter | str | ---- | same as above
| gain | float32 | ---- | same as above
| image_filename | str | ---- | same as above
| image_hdu | int16 | ---- | same as above
| nmatch | int16 | ---- | number of good detected sources with good PS1 (stars-photom.fits) or Gaia (stars-astrom.fits) 1'' matches
| ra | float64 | deg | right ascension of detected source
| x | float64 | pixel | x-position of detected source, [0,width]
| y | float64 | pixel | y-position of detected source, [0,height]

##### (3a) <image name>-star-photom.fits
The photom table has additional photometry information for 1 as matched PS1 sources

| Name | Type | Units | Description
| ------ | ------ | ------ | ------ |
| apflux | float64 | e- | number e- in 7'' aperture minus apskyflux 
| apmag | float64 | AB mag | apflux converted to AB Mag
| apskyflux | float64 | e- | number e- from local sky in 7'' aperture
| apskyflux_perpix | float64 | e-/pixel | number e- from local sky per pixel
| dmagall | float64 | AB mag | PS1 mag minus apmag
| gaia_g | float64 | AB mag | gaia g-band mag
| ps1_mag | float32 | AB mag | PS1 mag converted to either decam,mosaic,90prime system 
| ps1_g | float64 | AB mag | PS1 g-mag
| ps1_r | float64 | AB mag | PS1 r-mag
| ps1_i | float64 | AB mag | PS1 i-mag
| ps1_z | float64 |AB mag | PS1 z-mag

##### (3b) <image name>-star-astrom.fits
The astrom table has additional astrometry information for 1 as matched Gaia sources (or PS1 sources if < 20 Gaia sources).

| Name | Type | Units | Description
| ------ | ------ | ------ | ------ |
| decdiff | float64 | deg | "Gaia Dec" - dec
| radiff | float64 | deg | ("Gaia Ra" - ra) * cos(dec)


