import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import astropy.units as u

##### Define some needed auxilliary functions

## Convert from spherical to cartesian
def xyzsphere( theta, phi, r ):
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return x, y, z

## Rotate phi given an omega
def rot_phi( phi, w, t ):

    return ( phi + w * t )

# See if points are within a circle
def spherecircle( x, y, z, R, thc, phc, r, cl ):

    xc, yc, zc = xyzsphere( thc, phc, R )

    ds = np.sqrt( ( x0 - xc ) ** 2 + ( y0 - yc ) ** 2 + ( z0 - zc ) ** 2.0 )
    ds = R * np.arcsin( ds / ( 2 * R ** 2 ) * np.sqrt( 4 * R ** 2.0 - ds ** 2 ) )

    if cl == 'II': # Only one spot
        incircle = ( ds <= r ) & ( xc * x0 > 0 ) & ( yc * y0 > 0 ) & ( zc * z0 > 0 )
    elif cl == 'IIp': # Two opposite spots
        incircle = ds <= r
    
    return incircle

## Quadratic lim darkening 
def limbdark( x, z ):

    r2 = x ** 2.0 + z ** 2.0
    mu = np.sqrt( 1 - r2 )

    a = 0.7955
    b = -0.0714

    return 1 - a * ( 1 - mu ) - b * ( 1 - mu ) ** 2.0

# Adding noise to the light curve if wanted
def noise( TorF ):

    if TorF:
        return np.random.normal() * 5e-3
    else:
        return 0.0

##### Defining classes for the hotspots

## First Class IIp Hotspot
class ClassIIpHotspot():

    def __init__( self, N, trange ):

        self.N   = N # Number of points

        self.a   = np.random.uniform( 1.05, 1.5, N ) # Amplitude of spot
        self.t0  = np.random.uniform( trange[0], trange[1], N ) # t0 from a distribution
        self.t0  = np.zeros( N ) # t0 at 0
        self.dur = np.random.normal( 30.0, 5.0, N ) # duration

        # Picking north or south hemisphere
        i = np.random.uniform()
        if i < 0.5:
            self.thc = np.random.uniform( 0.0, np.pi/4, N )
        else:
            self.thc = np.random.uniform( 3*np.pi/4, 2*np.pi, N )

        self.phc = np.random.uniform( 0.0, 2 * np.pi, N ) # phi center
        self.rc  = np.random.uniform( 0.0, 0.3, N ) # size of spot
        
        return None

## Class II hotspot, same things as above
class ClassIIHotspot():

    def __init__( self, N, trange ):

        self.N   = N

        self.a   = np.random.uniform( 1.05, 1.3, N )
        self.t0  = np.random.uniform( trange[0], trange[1], N )
        self.t0  = trange[1] * np.random.random(N)
        self.dur = np.random.uniform( 1.0, 5.0, N )

        self.thc = np.random.normal( np.pi/2, np.pi/4, N )

        self.phc = np.random.uniform( 0.0, 2 * np.pi, N )
        self.rc  = np.random.uniform( 0.0, 0.3, N )

        self.tup = 0.3 * np.ones( N ) #rise time
        
        return None

## Function to calculate the flux
def getflux( t, flux, x, z, spotsII, spotsIIp, cont ):

    outflux = flux.copy()

    hitspot = False

    # Go through class II spots
    for i in range( spotsII.N ):

        # If during spot
        if spotsII.t0[i] <= t <= spotsII.t0[i] + spotsII.dur[i]:
            inspot = spherecircle( x0, y0, z0, R, spotsII.thc[i], spotsII.phc[i], spotsII.rc[i], 'II' )

            # Calculate profile
            if t <= spotsII.t0[i] + spotsII.tup[i]:
                att  = np.exp( np.log( spotsII.a[i] ) / spotsII.tup[i] * ( t - spotsII.t0[i] ) )
            else:
                att  = ( spotsII.a[i] - 1.0 ) * np.exp( -np.log( spotsII.a[i] ) / spotsII.tup[i] * ( t - ( spotsII.t0[i] + spotsII.tup[i] ) ) ) + 1.0

            fn = 1.0 + ( att * cont - cont ) / inspot.sum() # Flux set so it adds up to correct percentage

            if hitspot == False: outflux[inspot] = fn
            else: outflux[inspot] += fn
            hitspot = True

    # Go through Class IIp spots
    for i in range( spotsIIp.N ):

        if spotsIIp.t0[i] <= t <= spotsIIp.t0[i] + spotsIIp.dur[i]:
            inspot = spherecircle( x0, y0, z0, R, spotsIIp.thc[i], spotsIIp.phc[i], spotsIIp.rc[i], 'IIp' )

            fn = 1.0 + ( spotsIIp.a[i] * cont - cont ) / inspot.sum()

            if hitspot == False: outflux[inspot] = fn
            else: outflux[inspot] += fn
            hitspot = True

    outflux = outflux * limbdark( x, z )

    return outflux

###################################################################

# Create a sphere
R            = 1.0
tharr        = np.linspace( 0.0, np.pi, 500 )
pharr        = np.linspace( 0.0, 2 * np.pi, 500 )
tharr, pharr = np.meshgrid( tharr, pharr )
x0,y0,z0     = xyzsphere( tharr, pharr, R )

tmax = 30.0 # Duration of observation

# Set up class instances for spots -- set number of times and time range
hspotsII  = ClassIIHotspot( 24, [ 0.0, tmax ] )
hspotsIIp = ClassIIpHotspot( 2, [ 0.0, tmax ] )

# Set up continuum flux given number of points
flux = np.sin( tharr ) * np.sin( pharr )
cont = ( flux * limbdark( x0, z0 ) )[np.where( y0 > 0 )].sum()

# Stellar rotation speed (2pi/period)
omega = 2 * np.pi / 10

# Set number of points given cadence
cadence = 512.0 * u.s.to('d')
N       = int( tmax / cadence )

tarr = np.linspace( 0, tmax, N )
Farr = np.zeros( N )

for i in range( tarr.size ):
    print i
    
    newpharr = rot_phi( pharr, omega, tarr[i] )
    x,y,z    = xyzsphere( tharr, newpharr, R )

    toobs    = np.where( y > 0.0 )
    flux     = np.sin( tharr ) * np.sin( newpharr )

    f        = getflux( tarr[i], flux, x, z, hspotsII, hspotsIIp, cont )

    Farr[i]  = f[toobs].sum() + noise( True ) * cont

plt.clf()
plt.plot( tarr, Farr / cont, 'k-' )
plt.ylim( 0.5, 1.5 )
plt.show()
