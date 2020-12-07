def readDR10spec(input):
        """
        Load a 1D SDSS DR10 spectrum and extract wavelength solution.
        @param input: input file name
        @type input: string
        """
        import astropy.io.fits as pyfits
        import numpy as np
        dat = pyfits.open(input)
        wl = 10.0**(dat[1].data['loglam'])
        temperr = dat[1].data['ivar']
        air = np.mean(dat[2].data['airmass'])
        mask = temperr <= 0.0
        temperr[mask] = 1.0e-5
        err = 1./np.sqrt(temperr)
        return {'wl':wl,'flux':dat[1].data['flux'],'error':err,'z':dat[2].data['Z'],'air':air}

def stacker(z_dr12, plate, mjd, fiberid):
        """
        Median spectrum for a given set of spectra.
        nanmedian considers nan points
        flux is normalized to somewhere in continuim
        """
        
        from tqdm import tqdm
        import numpy as np
        import scipy
        ## Define a log wavelength grid for the composite spectrum
        step = 1.00015
        bb = np.arange(0,8813,1)
        wgrid = 800.0 * step**bb
        nw = len(bb)

        nqsos = len(z_dr12)
        #  initialize the spectrum with zeros
        sp = np.zeros([int(nqsos), nw])
        
        for i in tqdm(range(nqsos)):
        # Retrieve the spectra:
                file = 'data/spectra/%d/spec-%d-%d-%04d.fits' % (plate[i], plate[i],mjd[i],fiberid[i])
                spec = readDR10spec(file)
                wave = spec['wl']
                wz = wave/(z_dr12[i]+1)
                flux = spec['flux']
                mask = (wz > 1680.0) & (wz < 1730.0)
                fnorm = np.median(flux[mask])
                fluxn = flux/fnorm
        # interpolate the rest-frame spectrum onto the standard grid
                f = scipy.interpolate.interp1d(wz,fluxn,bounds_error=False,fill_value=float('nan'))
                sp[i] = f(wgrid)
        # calculate the median spectrum
        med1 = np.nanmedian(sp,axis=0)
        return med1

def ParStacker(z_dr12, plate, mjd, fiberid, ncores):
        """
        Median spectrum for a given set of spectra.
        nanmedian considers nan points
        flux is normalized to somewhere in continuim
        """
        
        from tqdm import tqdm
        import numpy as np
        import scipy
        ## Define a log wavelength grid for the composite spectrum
        step = 1.00015
        bb = np.arange(0,8813,1)
        wgrid = 800.0 * step**bb
        nw = len(bb)

        nqsos = len(z_dr12)
        #  initialize the spectrum with zeros
        sp = np.zeros([int(nqsos), nw])
        
        # Retrieve the spectra:
        def stacking(i):
                file = 'data/spectra/%d/spec-%d-%d-%04d.fits' % (plate[i], plate[i],mjd[i],fiberid[i])
                spec = readDR10spec(file)
                wave = spec['wl']
                wz = wave/(z_dr12[i]+1)
                flux = spec['flux']
                mask = (wz > 1680.0) & (wz < 1730.0)
                fnorm = np.median(flux[mask])
                fluxn = flux/fnorm
        # interpolate the rest-frame spectrum onto the standard grid
                f = scipy.interpolate.interp1d(wz,fluxn,bounds_error=False,fill_value=float('nan'))
                sp[i] = f(wgrid)
                
        from joblib import Parallel, delayed
        import multiprocessing
        # num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=ncores)(delayed(stacking)(i) for i in range(nqsos))

        # calculate the median spectrum
        med1 = np.nanmedian(sp,axis=0)
        return med1
