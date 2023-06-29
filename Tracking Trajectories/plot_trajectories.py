#This script is used to visualise the movement of the car based on its changing 
#coordinates which are captured through TeraTerm and saved on a '.txt' file. 
import numpy as np
import matplotlib.pyplot as plt


with open('C:/Users/prith/Desktop/lep.txt', 'r') as file:
    data = file.read()

# Parse the data
lines = data.strip().split('\n')
pos_data = np.genfromtxt(lines, delimiter=',', usecols=(3, 4))

# Extract x and y values
xs = pos_data[:, 0]
ys = pos_data[:, 1]


# Create the plot
plt.plot(xs, ys)
plt.title('Tracked Trajectory Positions of the Robot Car using the controller')
plt.xlabel('X- In Metres')
plt.ylabel('Y- In Metres')

# Add the diagonal line
x_line = np.linspace(min(xs), max(xs), 100)
y_line = np.linspace(min(ys), max(ys), 100)
plt.plot(x_line, y_line, '--', color='red')
plt.show()
