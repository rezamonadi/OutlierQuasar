# OutlierQuasar

## Download Data

https://data.sdss.org/sas/dr16/eboss/qso/DR16Q/DR16Q_v4.fits

## Data description
Here is a description for all of the rows in the table:

https://data.sdss.org/datamodel/files/BOSS_QSO/DR16Q/DR16Q_v4.html

## How to open fits files 

You can use topcat to open fits files:
http://www.star.bris.ac.uk/%7Embt/topcat/

But python can deal with fits files by astropy:
https://docs.astropy.org/en/stable/io/fits/

---
## Astropy
To install `astropy` you can simply run this if you haven't use `anaconda`:

    pip install --no-deps astropy
If you used `anaconda`, astropy is already installed and you may just run this:

    conda update astropy