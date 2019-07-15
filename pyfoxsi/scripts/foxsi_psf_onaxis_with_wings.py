import numpy as np
import matplotlib.pyplot as plt



def gauss2d( x, y , amplitude, xo, yo, sigma_x, sigma_y, theta):
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

def lorentz2d(x, y, amp, xc, yc, sigma_x, sigma_y):
    center = [xc, yc]
    sigma = [sigma_x, sigma_y]
    l = (amp/np.pi)  / ((1.0 +( (x - center[0])/sigma[0])**2) + (1.0 + ( (y - center[1])/sigma[1] )**2 ))
    lorentz_with_cutoff = l * (gauss2d(x, y, -1.0, center[0], center[1], 10.0, 10.0, 0.0) + 1.0)
    return lorentz_with_cutoff

def foxsi_psf(x, y, center=[0,0]):

    params = [0.7232072, 0.19533395, 0.09657606, 1.5420149, 1.12446209, 0.70267759, 1.49216343, 1.49216343, 4.13935008, 4.13935008, 9.00424196, 9.00424196,2.19809692, 2.19809692]
    amplitude = params[0:4]
    width_x = params[6:14:2]
    width_y = params[7:15:2]

    g1 = gauss2d(x, y, amplitude[0], center[0], center[1], width_x[0], width_y[0], 0)
    g2 = gauss2d(x, y, amplitude[1], center[0], center[1], width_x[1], width_y[1], 0)
    g3 = gauss2d(x, y, amplitude[2], center[0], center[1], width_x[2], width_y[2], 0)

    lor = lorentz2d(x, y, amplitude[3], center[0], center[1], width_x[3], width_y[3])
    return g1 + g2 + g3 + lor

x = np.linspace(-100, 100, 301)
y = foxsi_psf(0, x)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.plot(x, y)
ax.set_yscale('log')
ax.set_ylim(5, 1e-4)
ax.set_xlabel('X [arcsec]')
ax.set_ylabel('Normalized Amplitude')
plt.show()
