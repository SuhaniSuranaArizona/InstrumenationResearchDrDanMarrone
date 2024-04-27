mport numpy as np
import matplotlib.pyplot as plt
import math


def fitting(rotation_angle):
  original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
  original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
  original_y = original_y - 1.590
  rotated_x = []
  rotated_y = []
  rotation_angle = (rotation_angle * math.pi)/180
  for index in range (len(original_x)):
    new_x = (original_x[index] * math.cos(rotation_angle)) + (original_y[index] * math.sin(rotation_angle))
    rotated_x.append(new_x)
    new_y = -(original_x[index] * math.sin(rotation_angle)) + (original_y[index] * math.cos(rotation_angle))
    rotated_y.append(new_y)
  #plt.scatter(original_x, original_y)
  #plt.scatter(rotated_x, rotated_y)
  #plt.show
  return(rotated_x, rotated_y, original_x, original_y)

def fitting_equation(original_x, x0, y0, phi, f):
    phi = (phi * math.pi)/180
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    new_x = (original_x * cos_phi) + (original_y * sin_phi)
    new_y = -(original_x * sin_phi) + (original_y * cos_phi)
    return(new_y - (y0 + (new_x - x0)**2)/(4*f) )**2

def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0

from scipy.optimize import curve_fit
rotated_values = {}
def main():
  for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
  print(rotated_values)


original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[20, 100, 1])


x0, y0, f = popt

# Generate the fitted curve
x_fit = np.linspace(original_x.min(), original_x.max(), 100)
y_fit = fitting_equation(x_fit, x0, y0, f)

# Plotting

plt.scatter(original_x, original_y, label='Original Data')
plt.plot(x_fit, y_fit, color='red', label='Fitted Parabola')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


main()

"""**Revised Submission - Using Chi-Squared Technique**



"""

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt



for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt



for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum(((y_fit - rotated_y)**2)/rotated_y)
    chi_values.append(chi2_stat)
    angles.append(i)


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot



def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[20, 100, 1])

print(popt)
x0, y0, f = popt

i = 19.5
while (i <= 21.0):

    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum(((y_fit - rotated_y)**2)/rotated_y)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.1

'''
for i in range (18, 22, 1):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum(((y_fit - rotated_y)**2)/rotated_y)
    chi_values.append(chi2_stat)
    angles.append(i)
'''


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot



def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[20, 100, 1])

print(popt)
x0, y0, f = popt

i = 19.8
while (i <= 20.0):

    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum(((y_fit - rotated_y)**2)/rotated_y)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.01

'''
for i in range (18, 22, 1):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum(((y_fit - rotated_y)**2)/rotated_y)
    chi_values.append(chi2_stat)
    angles.append(i)
'''


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot



def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[20, 100, 1])

print(popt)
x0, y0, f = popt

i = 19.875
while (i <= 19.900):

    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum(((y_fit - rotated_y)**2)/rotated_y)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.001

'''
for i in range (18, 22, 1):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum(((y_fit - rotated_y)**2)/rotated_y)
    chi_values.append(chi2_stat)
    angles.append(i)
'''


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot



def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[20, 100, 1])

print(popt)
x0, y0, f = popt

i = 19.880
while (i <= 19.885):

    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum(((y_fit - rotated_y)**2)/rotated_y)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.0001

'''
for i in range (18, 22, 1):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum(((y_fit - rotated_y)**2)/rotated_y)
    chi_values.append(chi2_stat)
    angles.append(i)
'''


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot



def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[20, 100, 1])

print(popt)
x0, y0, f = popt

i = 19.882
while (i <= 19.883):

    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum(((y_fit - rotated_y)**2)/rotated_y)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.00001

'''
for i in range (18, 22, 1):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum(((y_fit - rotated_y)**2)/rotated_y)
    chi_values.append(chi2_stat)
    angles.append(i)
'''


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot








print(chi_values)
print(angles)
print(np.min(chi_values))

"""# **I got my tilt angle as 19.88204 degrees**
# x0 = 18.41366315 inches
# y0 = 3.36854885 inches
# f = 34.73563903 inches
"""

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (19, 21, 1):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 15.4])

print(popt)
x0, y0, f = popt



for i in range (19, 22, 1):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum(((y_fit - rotated_y)**2)/rotated_y)
    chi_values.append(chi2_stat)
    angles.append(i)


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

# Define the fitting equation for a rotated parabola
def fitting_equation_rotated(x, x0, y0, f, phi):
    phi_rad = np.deg2rad(phi)
    x_rotated = x * np.cos(phi_rad) + y0 * np.sin(phi_rad)
    y_rotated = -(x * np.sin(phi_rad)) + y0 * np.cos(phi_rad)
    return (-(x_rotated - x0)**2) / (4*f) + y_rotated
def fitting_equation_not_rotated(x, x0, y0, f, phi):
    return (-(x - x0)**2) / (4*f) + y0


# Sample data
original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5]) / 25.4 - 1.590

# Perform the curve fitting with initial guesses
p0 = [20, 100, 40, 45]  # Initial guesses for x0, y0, f, phi
popt, pcov = curve_fit(fitting_equation_rotated, original_x, original_y, p0=p0)

# Extract the optimized parameters
x0, y0, f, phi = popt
print(x0)
print(popt)

# Generate the fitted curve
#x_fit = np.linspace(original_x.min(), original_x.max(), 100)
x_fit = np.linspace(-60, 60, 100)
#x_fit = np.linspace(-60, original_x.max(), 100)
y_fit = fitting_equation_rotated(x_fit, x0, y0, f, phi)
y_fit_not_rotated = fitting_equation_rotated(x_fit, x0, y0, f, 0)
# Plotting
rotated_x, rotated_y, original_x, original_y = fitting(29.8)
plt.scatter(original_x, original_y, label='Original Data')
plt.scatter(rotated_x, rotated_y, label='Original Rotated Data')
plt.plot(x_fit, y_fit, color='red', label='Fitted Rotated Parabola')
plt.plot(( x_fit),   y_fit_not_rotated, color='blue', label='Fitted Not Rotated Parabola')
plt.plot(0, 0, 'k+')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.gca().set_aspect('equal')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

# Define the fitting equation for a rotated parabola
def fitting_equation_rotated(xy, x0, y0, f, phi):
    x = xy[0]
    y = xy[1]
    phi_rad = np.deg2rad(phi)
    x_rotated = x * np.cos(phi_rad) + y * np.sin(phi_rad)
    y_rotated = -(x * np.sin(phi_rad)) + y * np.cos(phi_rad)
    return (-(x_rotated - x0)**2) / (4*f) + y_rotated
def fitting_equation_not_rotated(x, x0, y0, f, phi):
    return (-(x - x0)**2) / (4*f) + y0


# Sample data
original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5]) / 25.4 - 1.590

# Perform the curve fitting with initial guesses
p0 = [20, 100, 40, 45]  # Initial guesses for x0, y0, f, phi
xy = [original_x, original_y]
popt, pcov = curve_fit(fitting_equation_rotated, original_x, original_y, p0=p0)

# Extract the optimized parameters
x0, y0, f, phi = popt
print(x0)
print(popt)

# Generate the fitted curve
#x_fit = np.linspace(original_x.min(), original_x.max(), 100)
x_fit = np.linspace(-60, 60, 100)
#x_fit = np.linspace(-60, original_x.max(), 100)
y_fit = fitting_equation_rotated(x_fit, x0, y0, f, phi)
y_fit_not_rotated = fitting_equation_rotated(x_fit, x0, y0, f, 0)
# Plotting
rotated_x, rotated_y, original_x, original_y = fitting(-phi)
plt.scatter(original_x, original_y, label='Original Data')
plt.scatter(rotated_x, rotated_y, label='Original Rotated Data')
plt.plot(x_fit, y_fit, color='red', label='Fitted Rotated Parabola')
plt.plot(x_fit, y_fit_not_rotated, color='blue', label='Fitted Not Rotated Parabola')
plt.plot(0, 0, 'k+')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.gca().set_aspect('equal')
plt.show()



###### DPM VERSION
def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt



for i in range (5, 45, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt



for i in range (27, 32, 1):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt


i = 29.7
while (i<=31):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.1
print(np.min(chi_values))
print(chi_values)
print(angles)

import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt


i = 29.75
while (i<=30):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.01
print(np.min(chi_values))
print(chi_values)
print(angles)

import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt


i = 29.83
while (i<=29.86):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.001
print(np.min(chi_values))
print(chi_values)
print(angles)

import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt


i = 29.844
while (i<=29.846):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.00001
print(np.min(chi_values))
print(chi_values)
print(angles[78])
print("All index value of 3 is: ", np.where(chi_values == np.min(chi_values))[0])
import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

"""# **I got my tilt angle as 19.88204 degrees**
# x0 = 18.41366315 inches
# y0 = 3.36854885 inches
# f = 34.73563903 inches
"""

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (19, 21, 1):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 1.590

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 15.4])

print(popt)
x0, y0, f = popt



for i in range (19, 22, 1):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum(((y_fit - rotated_y)**2)/rotated_y)
    chi_values.append(chi2_stat)
    angles.append(i)


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

# Define the fitting equation for a rotated parabola
def fitting_equation_rotated(x, x0, y0, f, phi):
    phi_rad = np.deg2rad(phi)
    x_rotated = x * np.cos(phi_rad) + y0 * np.sin(phi_rad)
    y_rotated = -(x * np.sin(phi_rad)) + y0 * np.cos(phi_rad)
    return (-(x_rotated - x0)**2) / (4*f) + y_rotated
def fitting_equation_not_rotated(x, x0, y0, f, phi):
    return (-(x - x0)**2) / (4*f) + y0


# Sample data
original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5]) / 25.4 - 1.590

# Perform the curve fitting with initial guesses
p0 = [20, 100, 40, 45]  # Initial guesses for x0, y0, f, phi
popt, pcov = curve_fit(fitting_equation_rotated, original_x, original_y, p0=p0)

# Extract the optimized parameters
x0, y0, f, phi = popt
print(x0)
print(popt)

# Generate the fitted curve
#x_fit = np.linspace(original_x.min(), original_x.max(), 100)
x_fit = np.linspace(-60, 60, 100)
#x_fit = np.linspace(-60, original_x.max(), 100)
y_fit = fitting_equation_rotated(x_fit, x0, y0, f, phi)
y_fit_not_rotated = fitting_equation_rotated(x_fit, x0, y0, f, 0)
# Plotting
rotated_x, rotated_y, original_x, original_y = fitting(-19.8)
plt.scatter(original_x, original_y, label='Original Data')
plt.scatter(rotated_x, rotated_y, label='Original Rotated Data')
plt.plot(x_fit, y_fit, color='red', label='Fitted Rotated Parabola')
plt.plot(( x_fit),   y_fit_not_rotated, color='blue', label='Fitted Not Rotated Parabola')
plt.plot(0, 0, 'k+')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.gca().set_aspect('equal')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

# Define the fitting equation for a rotated parabola
def fitting_equation_rotated(xy, x0, y0, f, phi):
    x = xy[0]
    y = xy[1]
    phi_rad = np.deg2rad(phi)
    x_rotated = x * np.cos(phi_rad) + y * np.sin(phi_rad)
    y_rotated = -(x * np.sin(phi_rad)) + y * np.cos(phi_rad)
    return (-(x_rotated - x0)**2) / (4*f) + y_rotated
def fitting_equation_not_rotated(x, x0, y0, f, phi):
    return (-(x - x0)**2) / (4*f) + y0


# Sample data
original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5]) / 25.4 - 1.590

# Perform the curve fitting with initial guesses
p0 = [20, 100, 40, 45]  # Initial guesses for x0, y0, f, phi
xy = [original_x, original_y]
popt, pcov = curve_fit(fitting_equation_rotated, original_x, original_y, p0=p0)

# Extract the optimized parameters
x0, y0, f, phi = popt
print(x0)
print(popt)

# Generate the fitted curve
#x_fit = np.linspace(original_x.min(), original_x.max(), 100)
x_fit = np.linspace(-60, 60, 100)
#x_fit = np.linspace(-60, original_x.max(), 100)
y_fit = fitting_equation_rotated(x_fit, x0, y0, f, phi)
y_fit_not_rotated = fitting_equation_rotated(x_fit, x0, y0, f, 0)
# Plotting
rotated_x, rotated_y, original_x, original_y = fitting(-phi)
plt.scatter(original_x, original_y, label='Original Data')
plt.scatter(rotated_x, rotated_y, label='Original Rotated Data')
plt.plot(x_fit, y_fit, color='red', label='Fitted Rotated Parabola')
plt.plot(x_fit, y_fit_not_rotated, color='blue', label='Fitted Not Rotated Parabola')
plt.plot(0, 0, 'k+')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.gca().set_aspect('equal')
plt.show()



###### DPM VERSION
def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 2.04

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt



for i in range (5, 45, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 2.04

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt



for i in range (27, 32, 1):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 2.04

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt


i = 29.0
while (i<=31):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.1
print(np.min(chi_values))
print(chi_values)
print(angles)

import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 2.04

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt


i = 29.75
while (i<=30):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.01
print(np.min(chi_values))
print(chi_values)
print(angles)

import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 2.04

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt


i = 29.83
while (i<=29.86):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.001
print(np.min(chi_values))
print(chi_values)
print(angles)

import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 2.04

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt


i = 29.844
while (i<=29.846):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.00001
print(np.min(chi_values))
print(chi_values)
print(angles[78])
print("All index value of 3 is: ", np.where(chi_values == np.min(chi_values))[0])
import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 2.04

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt


i = 29.7
while (i<=31):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.1
print(np.min(chi_values))
print(chi_values)
print(angles)

import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 2.04

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt


i = 29.6
while (i<=31):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.1


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot



def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 2.04

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt


i = 29.8
while (i<=30):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.001

print("All index value of 3 is: ", np.where(chi_values == np.min(chi_values))[0])
print(angles[8])


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

def fitting_equation_not_rotated(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0
def fitting_equation(x, x0, y0, f):
    return (-(x - x0)**2) / (4*f) + y0


from scipy.optimize import curve_fit
from scipy.stats import chisquare

rotated_values = {}
chi_values = []
angles = []
for i in range (5, 90, 5):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    #rotated_values["Angle " + str(i)] = [rotated_x, rotated_y]
#print(rotated_values)

original_x = np.array([6, 8, 10, 12, 14, 22, 24, 26, 28, 30, 32, 34, 36, 38])
original_y = np.array([95.5, 105.6, 114.1, 120.2, 124.1, 124, 120, 114.8, 108.2, 100.1, 91.5, 81.2, 69.7, 57.5])/25.4
original_y = original_y - 2.04

popt, pcov = curve_fit(fitting_equation, original_x, original_y, p0=[-0.5, -0.2, 25.4])

print(popt)
x0, y0, f = popt


i = 29.844
while (i<=29.846):
    rotated_x, rotated_y, original_x, original_y = fitting(i)
    popt, pcov = curve_fit(fitting_equation, rotated_x, rotated_y, p0=[-0.5, -0.2, 25.4])
    x0, y0, f = popt
    y_fit = fitting_equation_not_rotated(rotated_x, x0, y0, f)
    chi2_stat = np.sum((y_fit - rotated_y)**2)
    chi_values.append(chi2_stat)
    angles.append(i)
    i+=0.000001

print("All index value of 3 is: ", np.where(chi_values == np.min(chi_values))[0])
print(angles[8])


import matplotlib.pyplot as plt


# Plot the arrays
#plt.figure(figsize=(8, 6))   # Set the figure size
plt.plot(angles, chi_values)               # Plot y vs. x
plt.xlabel('Angles')              # Label for x-axis
plt.ylabel('Chi-Values SQUARED')              # Label for y-axis
plt.grid(True)               # Display grid
plt.show()                   # Show the plot

"""29.84400800000001 (Tilt Angle)


"""
