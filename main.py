import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, render_template, request, redirect
import numpy as np
import json
import os
import copy

port = os.environ.get('PORT', 7777)

##
# COLORSCALES = """aggrnyl     agsunset    blackbody   bluered     blues       blugrn      bluyl       brwnyl
# bugn        bupu        burg        burgyl      cividis     darkmint    electric    emrld
# gnbu        greens      greys       hot         inferno     jet         magenta     magma
# mint        orrd        oranges     oryel       peach       pinkyl      plasma      plotly3
# pubu        pubugn      purd        purp        purples     purpor      rainbow     rdbu
# rdpu        redor       reds        sunset      sunsetdark  teal        tealgrn     turbo
# viridis     ylgn        ylgnbu      ylorbr      ylorrd      algae       amp         deep
# dense       gray        haline      ice         matter      solar       speed       tempo
# thermal     turbid      armyrose    brbg        earth       fall        geyser      prgn
# piyg        picnic      portland    puor        rdgy        rdylbu      rdylgn      spectral
# tealrose    temps       tropic      balance     curl        delta       oxy         edge
# hsv         icefire     phase       twilight    mrybm       mygbm""".split()


#Helper functions

def evalfunc(x,y,func, **kwargs):
	"""Evaluate a scalar-valued 2d function given the inputs"""
	z = np.zeros( (len(x), len(y)) )
	for col,x in enumerate(x):
		z[:,col] = func(x,y, **kwargs)
	return z

def plot2dfunc(x,y,func,**kwargs):
	"""Plot a 3d surface for a 2d function"""
	z = evalfunc(x,y,func, **kwargs)
	fig = go.Figure(go.Surface(
		contours = {
			#"x": {"show": True, "start": 1.5, "end": 2, "size": 0.04, "color":"white"},
			#"z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
		},
		x = x,
		y = y,
		z = z))
	return fig

######################################################################
#psi and psi^2 for different scenarios
def psi_particleInBox(x, n, L=10):
	return np.sqrt((2/L))* ( (np.sin(n * np.pi/L * x)) )
def psiSquared_partcleInBox(x, n, L=10):
	return psi_particleInBox(x, n, L=L) ** 2
def psi_2dSquareWell(x,y,L=10,n_x=1,n_y=1):
	"""L: length of box
	n_x: Energy level in x direction
	n_y: Energy level in y
	"""
	return (2/L) * np.sin((n_x * np.pi * x)/L)  * np.sin((n_y * np.pi * y)/L)
def psiSquared_2dSquareWell(x,y,L=10,n_x=1,n_y=1):
	return psi_2dSquareWell(x,y,L=L,n_x=n_x,n_y=n_y) ** 2

######################################################################
#plotting functions
def plotPsi_particleInBox(n=1,L=10):
	"""L: length of box
	n: energy level
	"""
	x = np.linspace(0,L)
	fig = go.Figure( data=go.Scatter(x=x, y=psi_particleInBox(x, n, L=L)) )
	fig.update_layout(xaxis_title="Position",yaxis_title="Wave funtion")
	return pio.to_html(fig, full_html=False, include_plotlyjs=False)

def plotPsiSquared_particleInBox(n=1, L=10):
	x = np.linspace(0,L)
	fig = go.Figure( data=go.Scatter(x=x, y=psiSquared_partcleInBox(x, n)) )
	fig.update_layout(xaxis_title="Position", yaxis_title="Probability Density")
	return pio.to_html(fig, full_html=False, include_plotlyjs=False)

def plotPsi_2dSquareWell(n_x=1, n_y=1,L=10):
	x=np.linspace(0,L)
	y=np.linspace(0,L)
	fig = plot2dfunc(x,y,psi_2dSquareWell,n_x=n_x,n_y=n_y,L=L)
	fig.update_layout(
			scene = {
				"xaxis": {"nticks": 20},
				"yaxis": {"nticks": 20},
				"zaxis": {"nticks": 5},
				"aspectratio": {"x": 1, "y": 1, "z": 0.5},
				"zaxis": dict(title="Wavefunction")
				})
	return pio.to_html(fig, full_html=False, include_plotlyjs=False)

def plotPsiSquared_2dSquareWell(n_x=1, n_y=1,L=10):
	x=np.linspace(0,L)
	y=np.linspace(0,L)
	fig = plot2dfunc(x,y,psiSquared_2dSquareWell,n_x=n_x,n_y=n_y,L=L)
	fig.update_layout(
			scene = {
				"xaxis": {"nticks": 20},
				"yaxis": {"nticks": 20},
				"zaxis": {"nticks": 5},
				"aspectratio": {"x": 1, "y": 1, "z": 0.5},
				"zaxis": dict(title="Probability Density")
				},
			autosize=False,
			width=800,
			height=800,)
	return pio.to_html(fig, full_html=False, include_plotlyjs=False)

#####################################################################
# See quantum.ipynb

ONE_OVER_ROOT_PI: float = 1/np.sqrt(np.pi)
_RADIUS: int = 20


#helper functions
def cart_to_sph(x,y,z):
	r = np.sqrt(x**2 + y**2 + z**2)
	phi = np.arccos(z/r) # [0, pi]
	theta = np.arctan2(y,x) #longitudinal angle, measured from +ve x axis, [0, 2pi)
	return r, theta, phi

def sph_to_cart(r,theta,phi):
	x = r * np.sin(phi) * np.cos(theta)
	y = r * np.sin(phi) * np.sin(theta)
	z = r * np.cos(phi)
	return x, y, z

# (n, l) -> R_{n,l}
psiRadial = {
	(1, 0): (lambda r, rho, c0, Z: 2 * (c0**1.5) * np.exp(-rho)),
	(2, 1): (lambda r, rho, c0, Z: (3**0.5)/3 * ((0.5 * c0)**1.5) * rho * np.exp(-rho/2)),
	(2, 0): (lambda r, rho, c0, Z: 2 * ((0.5 * c0)**1.5) * (1 - 0.5*rho) * np.exp(-rho/2)),
	(3, 2): (lambda r, rho, c0, Z: (2/27)*(0.4**0.5) * ((c0/3)**1.5) * (rho**2) * np.exp(-rho/3)),
	(3, 1): (lambda r, rho, c0, Z: (4* 2**0.5)/3 * ((c0/3)**1.5) * rho * (1-rho/6) * np.exp(-rho/3)),
	(3, 0): (lambda r, rho, c0, Z: 2 * ((c0/3)**1.5) * (1 - (2*rho/3) + (2*(rho**2)/27)) * np.exp(-rho/3)),
	(4, 0): (lambda r, rho, c0, Z: 1/96 * (24 - 36*rho*0.5 + 12*(rho*0.5)**2 - (rho*0.5)**3) * (Z**1.5) * np.exp(-rho/4)),
	(4, 1): (lambda r, rho, c0, Z: 1/(32 * (15**0.5)) * rho*0.5 * (20-10*rho*0.5 + (rho*0.5)**2) * Z**1.5 * np.exp(-rho/4)), #unsure of beginning
	(4, 2): (lambda r, rho, c0, Z: 1/(96* 5**0.5) * (6-rho*0.5) * (rho*0.5)**2 * (Z**1.5) * np.exp(-rho/4)),
	(4, 3): (lambda r, rho, c0, Z: 1/(96* 35**0.5) * (rho*0.5)**3 * (Z**1.5) * np.exp(-rho/4))
}

# (l,m) -> Y_real{l}m
realSphereHarmonics = {
	(0, 0): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * 0.5),
	(1,-1): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * (0.75**0.5) * np.sin(theta) * np.sin(phi)),
	(1, 0): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * (0.75**0.5) * np.cos(phi)),
	(1, 1): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * (0.75**0.5) * np.sin(phi) * np.cos(theta)),
	(2,-2): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * 0.5 * (15**0.5) * (np.sin(phi)**2) * np.cos(theta) * np.sin(theta)),
	(2,-1): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * 0.5 * (15**0.5) * np.sin(phi) * np.cos(phi) * np.sin(theta)),
	(2, 0): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * 0.25 * (5**0.5) * (3 *(np.cos(phi)**2) - 1)),
	(2, 1): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * 0.5 * (15**0.5) * np.cos(phi) * np.sin(phi) * np.cos(theta)),
	(2, 2): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * 0.25 * (15**0.5) * (np.sin(phi)**2) * np.cos(2*theta)),
}

realSphereHarmonicsCart = {
	(3,-3): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * 0.25*(17.5**0.5) * (3*x**2 - y**2) * y/(r**3) ),
	(3,-2): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * 0.5 * (105**0.5) * x*y*z/(r**3)),
	(3,-1): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * 0.25 * (10.5**0.5) * y * (4*z**2 - x**2 - y**2)/(r**3) ),
	(3, 0): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * 0.25 * (7**0.5) * z * (2*z**2 -3*x**2 - 3*y**2)/(r**3) ),
	(3, 1): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * 0.25 * (10.5**0.5) * x * (4*z**2 - x**2 - y**2)/(r**3) ),
	(3, 2): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * 0.25 * (105**0.5) * (x**2 - y**2)*z/(r**3) ),
	(3, 3): (lambda theta, phi, x, y, z, r: ONE_OVER_ROOT_PI * 0.25*(17.5**0.5) * (x**2 - 3*y**2) * x/(r**3) ),
}

#(n, l, m) -> name
subshellNames = {
	(1, 0, 0): "1s",
	(2, 0, 0): "2s",
	(2, 1, 0): "2pz",
	(2, 1,-1): "2py",
	(2, 1, 1): "2px",
	(3, 0, 0): "3s",
	(3, 1, 0): "3pz",
	(3, 1,-1): "3py",
	(3, 1, 1): "3px",
	(3, 2, 0): "3dz^2",
	(3, 2,-1): "3dyz",
	(3, 2, 1): "3dxz",
	(3, 2,-2): "3dxy",
	(3, 2, 2): "3dx^2-y^2",
	(4, 0, 0): "4s",
	(4, 1, 0): "4pz",
	(4, 1,-1): "4py",
	(4, 1, 1): "4px",
	(4, 2, 0): "4dz^2",
	(4, 2,-2): "4dxy",
	(4, 2, 2): "4dx^2 - y^2",
	(4, 2,-1): "4dyz",
	(4, 2, 1): "4dxz",
	(4, 3, 0): "4fz^3",
	(4, 3,-3): "4fy(3x^2 - y^2)",
	(4, 3, 3): "4fx(x^2 - 3y^2)",
	(4, 3,-2): "4fxyz",
	(4, 3, 2): "4fz(x^2 - y^2)",
	(4, 3,-1): "4fyz^2",
	(4, 3, 1): "4fxz^2"
}

subshellLatex = {
	(1, 0, 0): "1s",
	(2, 0, 0): "2s",
	(2, 1, 0): "2p_z",
	(2, 1,-1): "2p_y",
	(2, 1, 1): "2p_x",
	(3, 0, 0): "3s",
	(3, 1, 0): "3p_z",
	(3, 1,-1): "3p_y",
	(3, 1, 1): "3p_x",
	(3, 2, 0): "3d_{z^2}",
	(3, 2,-1): r"3d_{yz}",
	(3, 2, 1): r"3d_{xz}",
	(3, 2,-2): r"3d_{xy}",
	(3, 2, 2): r"3d_{x^2-y^2}",
	(4, 0, 0): "4s",
	(4, 1, 0): "4p_z",
	(4, 1,-1): "4p_y",
	(4, 1, 1): "4p_x",
	(4, 2, 0): "4d_{z^2}",
	(4, 2,-2): r"4d_{xy}",
	(4, 2, 2): r"4d_{x^2 - y^2}",
	(4, 2,-1): r"4d_{yz}",
	(4, 2, 1): r"4d_{xz}",
	(4, 3, 0): r"4f_{z^3}",
	(4, 3,-3): r"4f_{y(3x^2 - y^2)}",
	(4, 3, 3): r"4f_{x(x^2 - 3y^2)}",
	(4, 3,-2): r"4f_{xyz}",
	(4, 3, 2): r"4f_{z(x^2 - y^2)}",
	(4, 3,-1): r"4f_{yz^2}",
	(4, 3, 1): r"4f_{xz^2}"
}



def psiSquared(psiR, psiY, r, theta, phi, rho, c0, Z, X_, Y_, Z_):
	return (
		psiR(r, rho, c0, Z) * psiY(theta, phi, X_, Y_, Z_, r)
	) ** 2

def scaledPsiSquared2(psiR, psiY, r, theta, phi, rho, c0, Z, X_, Y_, Z_):
	return psiSquared(psiR, psiY, r, theta, phi, rho, c0, Z, X_, Y_, Z_) * (200*r+1)

def scaledPsiSquared(psiR, psiY, r, theta, phi, rho, c0, Z, X_, Y_, Z_, scale_factor=1000):
	retval = (psiSquared(psiR, psiY, r, theta, phi, rho, c0, Z, X_, Y_, Z_)
		) * scale_factor
	return retval

def getProbsBetween(arr, minprob=0.5, maxprob=1):
	return np.array(list(map(lambda x: 1 if minprob<=x<=maxprob else 0, arr)))

def plotPsiFromRectangularCoords(n=1,l=0,m=0, Z=1, a0=1, isomax=1, opaque=False, whole=True, RADIUS=20, scale_factor=1000):
	"""n is the principal quantum number/energy level, at least 1
	l is the orbital angular momentum quantum number, from 0 to n-1
	m is the magnetic quantum number, from -l to l
	Z is the atomic number
	a0 is the Bohr radius
	"""
	assert (n > 0 and (0 <= l < n) and (abs(m) <= l))
	c0 = Z/a0

	if whole:
		X_, Y_, Z_ = np.mgrid[-RADIUS:RADIUS:40j, -RADIUS:RADIUS:40j, -RADIUS:RADIUS:40j]
	else:
		X_, Y_, Z_ = np.mgrid[0:2*RADIUS:40j, -RADIUS:RADIUS:40j, -RADIUS:RADIUS:40j]
	r, theta, phi = cart_to_sph(X_, Y_, Z_)

	rho = c0 * r

	psiR = psiRadial[(n, l)]
	if l >= 3:
		psiY = realSphereHarmonicsCart[(l,m)]
	else:
		psiY = realSphereHarmonics[(l,m)]

	print(subshellNames[(n, l, m)])
	psiSquaredValues = scaledPsiSquared(psiR, psiY, r, theta, phi, rho, c0, Z, X_, Y_,
			Z_, scale_factor=scale_factor).flatten()
	if opaque:
		fig = go.Figure(data=go.Isosurface(
			x=X_.flatten(),
			y=Y_.flatten(),
			z=Z_.flatten(),
			value=psiSquaredValues,
			isomin=0.004,
			isomax=isomax,
			colorscale="jet",
			opacity=1, # needs to be small to see through all surfaces
			surface_count=30, # needs to be a large number for good volume rendering
		))
	else:
		fig = go.Figure(data=go.Volume(
			x=X_.flatten(),
			y=Y_.flatten(),
			z=Z_.flatten(),
			value=psiSquaredValues,
			isomin=0,
			isomax=isomax,
			colorscale="jet",
			opacity=0.1,
			surface_count=30,
		))
	fig.update_layout(
		autosize=False,
	width=800,
	height=800,)
	return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def getOrbitalPlotstrings(n=1,l=0,m=0, opaque=False, whole=True):
	pref = f'{n}+{l}+{m}+'
	if opaque: pref += 'opq_'
	else: pref += 'trans_'
	if whole: pref += 'whole'
	else: pref += 'crs'
	pref += '.txt'
	with open(pref) as f:
		s = f.read()
	return s



app = Flask(__name__)

@app.route('/')
def home():
	s = plotPsi_2dSquareWell(n_x=2,n_y=2)
	return render_template('index.html', s=s, p2dstyle='display: block;', p1dstyle='display: none;', orbstyle='display: none;', curr_choice_ind=0)

@app.route('/orb', methods=["POST"])
def orbitals():
	n = int(request.form['n'])
	l = int(request.form['l'])
	m = int(request.form['m'])
	opaque = request.form['opq'] == 'opq'
	whole = request.form['whole'] == 'whole'
	s = getOrbitalPlotstrings(n=n,l=l,m=m,opaque=opaque,whole=whole) #plotPsiFromRectangularCoords(n=n, l=l, m=m, opaque=opaque, whole=whole, isomax=0.5)
	s = r"<div id='orbltx'> \[" + subshellLatex[(n,l,m)] + rf" \ orbital (n={n}, l={l}, m={m}) \] </div> <br>" + s
	s += "<br> <p>The value represented by the colour is drectly related to the probability density function of an electron in the atom.</p> <p>The origin represents the nucleus.</p> <p>Some detail might be difficult to make out; I suggest viewing both the opaque and transparent versions.</p> <p>The red surfaces and regions enclosed by them are areas of high probability density. The opposite is true for the blue regions.</p>"
	return render_template('index.html', s=s, p2dstyle='display: none;', p1dstyle='display: none;', orbstyle='display: block;', curr_choice_ind=2)

@app.route('/2dsquarewell',methods=['POST'])
def particle2DBox():
	nx = int(request.form['nx'])
	ny = int(request.form['ny'])
	func = request.form['psi']
	if func == 'psi':
		s = plotPsi_2dSquareWell(n_x=nx, n_y=ny)
	else:
		s = plotPsiSquared_2dSquareWell(n_x=nx,n_y=ny)
	return render_template('index.html', s=s, p2dstyle='display: block;', p1dstyle='display: none;', orbstyle='display: none;', curr_choice_ind=0)

@app.route('/1dsquarewell', methods=['POST'])
def particle1DBox():
	n = int(request.form['n'])
	func = request.form['psi']
	if func == 'psi':
		s = plotPsi_particleInBox(n=n)
	else:
		s = plotPsiSquared_particleInBox(n=n)
	return render_template('index.html', s=s, p2dstyle='display: none;', p1dstyle='display: block;', orbstyle='display: none;', curr_choice_ind=1)



#n, l-> Radius, Isomax, Scale factor
# d = {
# 	(1,0): (10, 1, 1000),
# 	(2,0): (20, 0.7, 1000),
# 	(2,1): (10, 1, 1000),
# 	(3,0): (10,0.5, 100),
# 	(3,1): (20,1, 100),
# 	(3,2): (20,0.5, 1000),
# 	(4,0): (20, 0.4, 1000),
# 	(4,1): (20,0.4, 1000),
# 	(4,2): (20,0.4, 1000),
# 	(4,3): (20,0.4, 1000)
# }

# dic = {}
# for n in range(1, 5):
# 	for l in range(n):
# 		for m in range(-l, l+1):
# 			s_trans_crs = plotPsiFromRectangularCoords(n=n, l=l, m=m, opaque=False, whole=False, isomax=d[(n,l)][1], RADIUS=d[(n,l)][0], scale_factor=d[(n,l)][2])
# 			s_opq_crs = plotPsiFromRectangularCoords(n=n, l=l, m=m, opaque=True, whole=False, isomax=d[(n,l)][1], RADIUS=d[(n,l)][0], scale_factor=d[(n,l)][2])
# 			s_trans_whole = plotPsiFromRectangularCoords(n=n, l=l, m=m, opaque=False, whole=True, isomax=d[(n,l)][1], RADIUS=d[(n,l)][0], scale_factor=d[(n,l)][2])
# 			s_opq_whole = plotPsiFromRectangularCoords(n=n, l=l, m=m, opaque=True, whole=True, isomax=d[(n,l)][1], RADIUS=d[(n,l)][0], scale_factor=d[(n,l)][2])

# 			dic[f"{n} {l} {m}"] = {"trans_crs": s_trans_crs, "opq_crs": s_opq_crs, "trans_whole": s_trans_whole, "opq_whole": s_opq_whole}

# with open('orbstrings.json', 'w') as f:
# 	json.dump(dic, f, indent=0)


# import re
# pat = r'([^A-Za-z0-9])\s+'
# for _dname in dic:
# 	for key in dic[_dname]:
# 		if 'tran_' in key:
# 			suffname = 'trans_' + key[key.find('_')+1:]
# 		else: suffname = key
# 		filename = f'{_dname} {suffname}.txt'.replace(' ', '+')
# 		with open(filename, 'w') as f:
# 			f.write(re.sub(pat, r'\1', dic[_dname][key]) )




"""
With scale factor 1000, RADIUS = 20
4s seems to work here too
4p orbitals: isomax: 0.4, isomin: 0, 0.004, might need slider for radius as some detail is not resolved
4d orbitals
4f orbitals
Uses psi^2

Scale factor 100, RADIUS = 20, psi^2
3p orbitals: 0-1

Scale factor 100, RADIUS = 10, psi^2
3s orbital: 0-0.5

Scale factor 1000, RADIUS = 20, psi^2
3d orbitals: 0-0.5
2s orbital: 0-0.7

Scale factor 1000, RADIUS = 10, psi^2
2p orbitals: 0-1
1s orbital: 0-1
"""
if __name__ == '__main__':
	app.run(port=port, threaded=True)

