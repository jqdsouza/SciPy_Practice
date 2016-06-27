# Fit a polynomial curve to a given set of data points using optimization

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def error_poly(C, data):
	"""Compute error between given polynomial and observed data

	Parameters
	----------
	C: numpy.poly1d object or equivalent array representing polynomial coefficients
	data: 2D array where each row is a point (x, y)

	Returns error as a single real value
	"""

	# Metric: Sum of squared Y-axis differences
	error = np.sum((data[:, 1] - np.polyval(C, data[:, 0])) ** 2)

	return error

def fit_poly(data, error_func, degree=3):
	"""Fit a polynomial to given data, using supplied error function

	Parameters
	----------
	data: 2D array where each row is a point (x, y)
	error_func: function that computes the error between a polynomial and observed data

	Returns polynomial that minimizes error function
	"""

	# Generate initial guess for polynomial model (all coeffs = 1)
	Cguess = np.poly1d(np.ones(degree + 1, dtype=np.float32))

	# Plot initial guess (optional)
	x = np.linspace(-5, 5, 21)
	plt.plot(x, np.polyval(Cguess, x), 'm--', linewidth=2.0, label="Initial guess")

	# Call optimizer to minimize error function
	result = spo.minimize(error_func, Cguess, args=(data,), method='SLSQP', options={'disp': True})

	return np.poly1d(result.x) # convert optimal result into a poly1d object


def test_run():
	# Define original line
	l_orig = np.float32([4, 2, 6]) 
	print "Original line: C0 = {}, C1 = {}".format(l_orig[0], l_orig[1])

	Xorig = np.linspace(0, 10, 21)
	Yorig = l_orig[0] * Xorig + l_orig[1]

	plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")

	# Generate noisy data points
	noise_sigma = 3.0
	noise = np.random.normal(0, noise_sigma, Yorig.shape)
	data = np.asarray([Xorig, Yorig + noise]).T

	plt.plot(data[:,0], data[:, 1], 'go', label="Data points")

	# Try to fit polynomial curve to this data
	l_fit = fit_poly(data, error_poly)
	print "Fitted line: C0 = {}, C1 = {}".format(l_fit[0], l_fit[1])

	plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], 'r--', linewidth=2.0)

	plt.show()

if __name__ == "__main__":
	test_run()

