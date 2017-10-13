# What Legacyzpts Creates
Four FITS data tables with the same columns and units for all three cameras.

### (1) <image name>-zpt.fits
Most important:

| Name | Type | Units | Description
| ------ | ------ | ------ | ------ |
| airmass | float32 | 
| camera | str |
| ccdname | str |
| date_obs | str |
| dec | float64 | 
| decoff | float32 | 
| decrms | float32 |
| expnum | int32 | 
| err_message | str | 
| exptime | float32 | 
| filter | str | 
| fwhm | float32 | 
| fwhm_cp | float32 | 
| gain | float32 | 
| ha | str |
| height | int16 |
| image_filename | str | 
| mjd_obs | float64 | 
| nmatch_astrom | int16 | 
| nmatch_photom | int16 | 
| phoff | float32 | 
| phrms | float32 |
| pixscale | float32 | 
| ra | float64 | 
| raoff | float32 | 
| rarms | float32 | 
| skycounts | float32 | 
| skymag | float32 | 
| skyrms | float32 |
| transp | float32 |
| width | int16 | 
| zpt | float32 | 
| zptavg | float32 | 
 
Less important:

| Name | Type | Units | Description
| ------ | ------ | ------ | ------ |
| ccdnum | int16 | 
| cd1_1 | float32 | 
| cd1_2 | float32 | 
| cd2_1 | float32 | 
| cd2_2 | float32 |
| crpix1 | float32 | 
| crpix2 | float32 | 
| crval1 | float64 | 
| crval2 | float64 | 
| dec_bore | float64 | 
| decstddev | float32 |
| expid | str | 
| image_hdu | int16 | 
| mdncol | float32 |
| object | str | 
| propid | str | 
| ra_bore | float64 | 
| rastddev | float32 | 
| ut | str | 

### (2) <image name>-legacyzpt.fits
All of these are duplicates of #1 and are the minimal set required by our legacypipe/Tractor pipeline. Note, the names and units are different from above.

| Name | Type | Units | Description
| ------ | ------ | ------ | ------ |
| camera | str | 
| ccddecoff | float32 | 
| ccdname | str |
| ccdnmatch | int16 | 
| ccdraoff | float32 | 
| ccdzpt | float32 |
| cd1_1 | float32 | 
| cd1_2 | float32 | 
| cd2_1 | float32 | 
| cd2_2 | float32 | 
| crpix1 | float32 | 
| crpix2 | float32 | 
| crval1 | float64 | 
| crval2 | float64 |
| dec | float64 |
| dec_bore | float64 |
| expnum | int32 | 
| exptime | float32 | 
| filter | str |
| fwhm | float32 | 
| height | int16 | 
| image_filename | str | 
| image_hdu | int16 | 
| mjd_obs | float64 | 
| object | str | 
| propid | str | 
| ra | float64 | 
| ra_bore | float64 | 
| skyrms | float32 | 
| width | int16 | 
| zpt | float32 | 

### (3) <image name>-star-photom.fits

| Name | Type | Units | Description
| ------ | ------ | ------ | ------ |
| amplifier | int16 | 
| apflux | float64 | 
| apmag | float64 | 
| apskyflux | float64 | 
| apskyflux_perpix | float64 | 
| ccdname | str | 
| daofind_x | float32 | 
| daofind_y | float32 | 
| dec | float64 |
| decdiff | float64 |
| decdiff_ps1 | float64 |
| dmagall | float64 | 
| expnum | int32 | 
| expid | str | 
| exptime | float32 | 
| filter | str |
| gain | float32 | 
| gaia_dec | float64 |
| gaia_g | float64 | 
| gaia_ra | float64 | 
| image_filename | str | 
| image_hdu | int16 | 
| mycuts_x | float32 | 
| mycuts_y | float32 | 
| nmatch | int16 | 
| ps1_gicolor | float32 | 
| ps1_mag | float32 | 
| ps1_g | float64 | 
| ps1_r | float64 | 
| ps1_i | float64 | 
| ps1_z | float64 |
| ra | float64 |
| radiff | float64 | 
| radiff_ps1 | float64 | 
| x | float64 | 
| y | float64 | 

### (4) <image name>-star-astrom.fits
This table has most of the above plus these astrometric qauntities

| Name | Type | Units | Description
| ------ | ------ | ------ | ------ |
| decdiff | float64 |
| decdiff_ps1 | float64 |
| nmatch | int16 | 
| radiff | float64 | 
| radiff_ps1 | float64 | 
| x | float64 | 
| y | float64 | 


