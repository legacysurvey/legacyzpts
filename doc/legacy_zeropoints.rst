=================
legacy_zeropoints
=================

Generate a legacypipe-compatible CCD-level zeropoints file for a given set of (reduced) BASS, MzLS, or DECaLS imaging.

Inputs
======

CP processed imaging in 
/project/projectdirs/cosmo/staging/{decam,mosaicz,bok}
and
/global/projecta/projectdirs/cosmo/staging/{decam,bok}


Output Files
============

/global/cscratch1/sd/kaylanb/dr5_zpts/{decam,mosaic,90prime}

There are two outputs for every CP image:
* zeropoints table: "<image_name>-zpt.fits"
* stars table: "<image_name>-star.fits"

These have the same units independent of camera. The original
zeropoints and stars files created by 
"decstat.pro, mosstat.pro, bokstat.pro" had different units depending
on the camera. The above zeropoints and stars tables can be 
converted to the same units and column names as the IDL code's 
tables. 

This is what the tests/ do. Making sure legacyzpts reproduces the
original IDL versions.


Notes
=====

This script borrows liberally from code written by Ian, Kaylan, Dustin, David
S. and Arjun, including rapala.survey.bass_ccds, legacypipe.simple-bok-ccds,
obsbot.measure_raw, and the IDL codes decstat and mosstat.

Although the script was developed to run on the temporarily repackaged BASS data
created by the script legacyccds/repackage-bass.py (which writes out
multi-extension FITS files with a different naming convention relative to what
NAOC delivers), it is largely camera-agnostic, and should therefore eventually
be able to be used to derive zeropoints for all the Legacy Survey imaging.

On edison the repackaged BASS data are located in
/scratch2/scratchdirs/ioannis/bok-reduced with the correct permissions.

Proposed changes to the -ccds.fits file used by legacypipe:
 * Rename arawgain --> gain to be camera-agnostic.
 * The quantities ccdzpta and ccdzptb are specific to DECam, while for 90prime
   these quantities are ccdzpt1, ccdzpt2, ccdzpt3, and ccdzpt4.  These columns
   can be kept in the -zeropoints.fits file but should be removed from the final
   -ccds.fits file.
 * The pipeline uses the SE-measured FWHM (FWHM, pixels) to do source detection
   and to estimate the depth, instead of SEEING (FWHM, arcsec), which is
   measured by decstat in the case of DECam.  We should remove our dependence on
   SExtractor and simply use the seeing/fwhm estimate measured by us (e.g., this
   code).
 * The pixel scale should be added to the output file, although it can be gotten
   from the CD matrix.
 * AVSKY should be converted to electron or electron/s, to account for
   the varying gain of the amplifiers.  It's actually not even clear
   we need this header keyword.
 * We probably shouldn't cross-match against the tiles file in this code (but
   instead in something like merge-zeropoints), but what else from the annotated
   CCDs file should be directly calculated and stored here?
 * Are ccdnum and image_hdu redundant?

