
import numpy as np

import scipy.signal as signal

# modified by DmBorisov
def sbandpass(s, h, freqlo, freqhi):
    nr = h.nr
    dt = h.dt
    fs = 1/dt
    for ir in range(0, nr):
        s[:,ir] = bandpass2(s[:,ir],freqlo,freqhi,fs) 

    return s


def sconvolve(s, h, w, inplace=True):
    nt = h.nt
    nr = h.nr

    if inplace:
        for ir in range(nr):
            s[:,ir] = np.convolve(s[:,ir], w, 'same')
        return s
    else:
        s2 = np.zeros((nt,nr))
        for ir in range(nr):
            s2[:,ir] = np.convolve(s[:,ir], w, 'same')
        return s2


def slowpass(s, h, freq):
    raise NotImplementedError


def shighpass(s, h, freq):
    raise NotImplementedError


# mute early arrivals (modified by DmBorisov)
def smute(s, h, vel, toff, xoff, mute_out, constant_spacing=False):
    nt = h.nt
    dt = h.dt
    nr = h.nr

    # construct tapered window
    length = 400
    win = np.sin(np.linspace(0, np.pi, 2*length))
    win = win[0:length]

    for ir in range(0, nr):

        # calculate slope
        if vel!=0:
            slope = 1./vel
        else:
            slope = 0

        # calculate offsets
        if constant_spacing:
            itoff = toff/dt
            ixoff = (ir-xoff)/dt
        else:
            itoff = toff/dt
            #ixoff = (h.rx[ir]-h.sx[0]-xoff)/dt
            ixoff = np.sqrt((h.rx[ir]-h.sx[0])**2 + (h.ry[ir]-h.sy[0])**2)/dt

        itmin = int(np.ceil(slope*abs(ixoff)+itoff)) - length/2
        itmax = itmin + length

        # apply window
        if 1 < itmin and itmax < nt:
            s[0:itmin,ir] = 0.
            s[itmin:itmax,ir] = win*s[itmin:itmax,ir]
        elif itmin < 1 <= itmax:
            s[0:itmax,ir] = win[length-itmax:length]*s[0:itmax,ir] 
        elif itmin < nt < itmax:
            s[0:itmin,ir] = 0.
            s[itmin:nt,ir] = win[0:nt-itmin]*s[itmin:nt,ir] 
        elif itmin > nt:
            s[:,ir] = 0.

        # inner mute 
        if ixoff*dt < xoff:
            s[:,ir] = 0.

        # outer mute 
        if ixoff*dt > mute_out:
            s[:,ir] = 0.

        # top mute
        s[0:length,ir] = win*s[0:length,ir]

        # bot mute
        s[nt-length:nt,ir] = win[::-1]*s[nt-length:nt,ir]

    return s


# mute late arrivals (added by DmBorisov)
def smutelow(s, h, vel, toff, xoff, mute_out, constant_spacing=False):
    nt = h.nt
    dt = h.dt
    nr = h.nr

    # construct tapered window
    length = 400
    win = np.cos(np.linspace(0, np.pi, 2*length))
    win = win[0:length]

    for ir in range(0,h.nr):
        # calculate slope
        if vel!=0:
            slope = 1./vel
        else:
            slope = 0

        # calculate offsets
        if constant_spacing:
            itoff = toff/dt
        else:
            itoff = toff/dt
            ixoff = np.sqrt((h.rx[ir]-h.sx[0])**2 + (h.ry[ir]-h.sy[0])**2)/dt

        itmin = int(np.ceil(slope*abs(ixoff)+itoff)) - length/2
        itmax = itmin + length

        # apply window
        if 1 < itmin and itmax < nt:
            s[itmax:nt,ir] = 0.
            s[itmin:itmax,ir] = win*s[itmin:itmax,ir]
        elif itmin < 1 <= itmax:
            s[0:itmax,ir] = win[length-itmax:length]*s[0:itmax,ir]
        elif itmin < nt < itmax:
            s[itmin:nt,ir] = win[0:nt-itmin]*s[itmin:nt,ir]

        # inner mute
        if ixoff*dt < xoff:
            s[:,ir] = 0.

        # outer mute 
        if ixoff*dt > mute_out:
            s[:,ir] = 0.

    return s


def swindow(s, h, tmin, tmax, alpha=0.05, units='samples'):
    nt = h.nt
    dt = h.dt
    nr = h.nr
    t0 = h.t0

    if units == 'time':
        itmin = int((tmin-t0)/dt)
        itmax = int((tmax-t0)/dt)

    elif units == 'samples':
        itmin = int(tmin)
        itmax = int(tmax)

    else:
        raise ValueError

    win = tukeywin(nt, itmin, itmax, alpha)
    for ir in range(0,nr):
        s[:,ir] = win*s[:,ir]

    return s


# --

def correlate(u, v):
    w = np.convolve(u, np.flipud(v))
    return


def bandpass(w, freqlo, freqhi, fs, npass=2):
    wn = [2*freqlo/fs, 2*freqhi/fs]
    (b,a) = signal.butter(npass, wn, btype='band')
    w = signal.lfilter(b, a, w)
    return w


# A forward-backward linear filter (filtfilt) (added by DmBorisov) 
def bandpass2(w, freqlo, freqhi, fs, npass=3):
    wn = [2*freqlo/fs, 2*freqhi/fs]
    (b,a) = signal.butter(npass, wn, 'bandpass')
    w = signal.filtfilt(b, a, w)
    return w


def highpass(w, freq, df, npass=2):
    raise Exception('Not yet implemented.')


def lowpass(w, freq, df, npass=2):
    raise Exception('Not yet implemented.')


def window(nt, type='sine', **kwargs):
    if type == 'sine':
        return np.sin(np.linspace(0, 1, 2*length))
    elif type == 'tukey':
        return tukeywin


def tukeywin(nt, imin, imax, alpha=0.05):
    t = np.linspace(0,1,imax-imin)
    w = np.zeros(imax-imin)
    p = alpha/2.
    lo = np.floor(p*(imax-imin-1))+1
    hi = imax-imin-lo
    w[:lo] = (1+np.cos(np.pi/p*(t[:lo]-p)))/2
    w[lo:hi] = np.ones((hi-lo))
    w[hi:] = (1+np.cos(np.pi/p*(t[hi:]-p)))/2
    win = np.zeros(nt)
    win[imin:imax] = w
    return win

