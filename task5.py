import numpy as np
import matplotlib.pyplot as plt
import sensor_fusion as sf
import lsqSolve as lsqS
import pandas as pd

# Helper functions
def dist_(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.hypot(x1 - x2, y1 - y2)

def phi_(x1, y1, x2, y2, psi):
    """
    Calculate the bearing angle from the robot to a QR code, relative to the robot's heading.
    Angles are in radians.
    """
    return np.arctan2(y2 - y1, x2 - x1) - psi

def plot_estimation_history(x_ls, J_history=None):
    RAD_TO_DEG = 180.0 / np.pi
    final_estimate = x_ls[:, -1]
    if J_history is not None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[1, 1].axis('off')
    axs[0, 0].plot(x_ls[0, :], label='Final X: {:.2f} cm'.format(final_estimate[0]))
    axs[0, 0].set_title('X Position over Iterations')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('X Position (cm)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[1, 0].plot(x_ls[1, :], label='Final Y: {:.2f} cm'.format(final_estimate[1]))
    axs[1, 0].set_title('Y Position over Iterations')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Y Position (cm)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[0, 1].plot(x_ls[2, :] * RAD_TO_DEG, label='Final Orientation: {:.2f}°'.format(final_estimate[2] * RAD_TO_DEG))
    axs[0, 1].set_title('Orientation over Iterations')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Orientation (degrees)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    if J_history is not None:
        axs[1, 1].plot(J_history, label='Cost Function')
        axs[1, 1].set_title('Cost Function over Iterations')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Cost')
        axs[1, 1].grid(True)
        axs[1, 1].legend()
    else:
        axs[1, 1].axis('off')
    plt.tight_layout()
    plt.show()

def plot_robot_position(x_true, x_estimated, qr_positions):
    """Plot the true and estimated robot positions along with QR code positions."""
    plt.figure(figsize=(10, 8))

    # Plot QR codes
    plt.scatter(qr_positions[:, 0], qr_positions[:, 1], 
                color='blue', marker='s', s=100, label='QR Codes')

    # Plot true robot position
    plt.scatter(x_true[0], x_true[1],
                color='green', marker='o', s=200, label='True Position')

    # True position heading arrow
    arrow_length = 5
    dx = arrow_length * np.cos(x_true[2])
    dy = arrow_length * np.sin(x_true[2])
    plt.arrow(x_true[0], x_true[1], dx, dy, head_width=1, head_length=1, fc='green', ec='green')

    # Plot estimated robot position
    plt.scatter(x_estimated[0], x_estimated[1],
                color='red', marker='o', s=200, label='Estimated Position')

    # Estimated position heading arrow
    dx = arrow_length * np.cos(x_estimated[2])
    dy = arrow_length * np.sin(x_estimated[2])
    plt.arrow(x_estimated[0], x_estimated[1], dx, dy, head_width=1, head_length=1, fc='red', ec='red')

    plt.xlabel('X Position (cm)')
    plt.ylabel('Y Position (cm)')
    plt.title('Robot Localization Results')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

QR_CODE_SIDE_LENGTH = 11.5  
FOCAL_LENGTH = 546.54 

camera_file = 'dataset1_part2/data/task5/camera_localization_task5.csv'
qr_code_file = 'dataset1_part2/data/qr_code_position_in_global_coordinate.csv'

# Read QR code positions
qr_code_data = pd.read_csv(qr_code_file)
qr_code_positions = {}
for _, row in qr_code_data.iterrows():
    qr_code_positions[int(row['qr_code'])] = np.array([row['mid_point_x_cm'], row['mid_point_y_cm']])

# Initialize the camera sensor
Camera = sf.Sensor('Camera', sf.CAMERA_COLUMNS, meas_record_file=camera_file, is_linear=False, start_index=0)

# True and initial positions (convert headings to radians)
x_true = np.array([60.9, 35.0, np.radians(90)])  # True position (x, y, heading in radians)
x_init = np.array([0.0, 0.0, np.radians(0)])  # Initial guess for optimization

distance_variance = (2.0)**2  # Variance of distance measurements in cm^2
angle_variance = (np.radians(5.0))**2  # Variance of angle measurements in radians^2
R_one_diag = np.array([distance_variance, angle_variance])

# Parameters for least squares optimization
I_max = 100  # Maximum number of iterations
gamma = 1  # Step size scaling parameter (can adjust if needed)
params_LSQ = {
    'x_sensors': None,  # To be set later with QR code positions
    'R': None,
    'LR': None,  # Cholesky factorization of the covariance matrix
    'Rinv': None,
    'gamma': gamma,
    'I_max': I_max,
    'Line_search': False,  # Set to True if you want to enable line search
    'Line_search_n_points': 10,
    'Jwls': lsqS.Jwls  # Cost function for weighted least squares
}

# Reset the camera sampling index
Camera.reset_sampling_index()

# Get the raw measurements from the camera
y_raw = Camera.get_measurement()

# Extract relevant data from the measurements
weight = y_raw[:, 3]
height = y_raw[:, 4]
c_x = y_raw[:, 1]

# Compute distances and angles from camera measurements
dist = QR_CODE_SIDE_LENGTH * FOCAL_LENGTH / height  # Distance to QR code in cm
direct = np.arctan2(c_x, FOCAL_LENGTH)  # Bearing angle in radians
angle_qr = np.arccos(np.minimum(weight, height) / height)  # QR code angle

# Correct distances (accounting for camera perspective)
corrected_dist = dist / np.cos(direct) + 0.5 * QR_CODE_SIDE_LENGTH * np.sin(angle_qr)
y_raw[:, 5] = corrected_dist
n_qr_codes = y_raw.shape[0]

# Build measurement vector y (distances and angles in radians)
y = np.zeros(2 * n_qr_codes)
for i in range(n_qr_codes):
    y[2*i] = y_raw[i, 5]  # Corrected distance in cm
    y[2*i + 1] = direct[i]  # Bearing angle in radians

# Map QR code IDs to their global positions
qr_pos = np.array([qr_code_positions[int(qr_id)] for qr_id in y_raw[:, 0].astype('int')])
params_LSQ['x_sensors'] = qr_pos

# Construct the covariance matrix
R = np.diag(np.kron(np.ones(n_qr_codes), R_one_diag))
params_LSQ['R'] = R
params_LSQ['LR'] = np.linalg.cholesky(R).T
params_LSQ['Rinv'] = np.linalg.inv(R)

#measurement function h_cam
def h_cam(x, params):
    x_sensors = params['x_sensors']  #x, y positions of QR codes
    h = np.zeros(2 * x_sensors.shape[0])
    x_c = x[0]
    y_c = x[1]
    psi = x[2]  
    for i in range(x_sensors.shape[0]):
        h[2*i] = dist_(x_c, y_c, x_sensors[i, 0], x_sensors[i, 1])  
        h[2*i + 1] = phi_(x_c, y_c, x_sensors[i, 0], x_sensors[i, 1], psi) 
    return h

#Jacobian of the measurement function H_cam
def H_cam(x, params):
    x_sensors = params['x_sensors']  #x, y positions of QR codes
    H = np.zeros((2 * x_sensors.shape[0], 3))
    x_c = x[0]
    y_c = x[1]
    for i in range(x_sensors.shape[0]):
        delta_x = x_c - x_sensors[i, 0]
        delta_y = y_c - x_sensors[i, 1]
        dist = dist_(x_c, y_c, x_sensors[i, 0], x_sensors[i, 1])
        if dist == 0:
            continue 
        H[2*i, :] = np.array([delta_x / dist, delta_y / dist, 0])
        H[2*i + 1, :] = np.array([delta_y / (dist**2), -delta_x / (dist**2), -1])
    return H

# Perform least squares optimization using Gauss-Newton method
xhat_history_GN, J_history_GN = lsqS.lsqsolve(
    y, h_cam, H_cam, x_init, params_LSQ, method='gauss-newton'
)

# Extract the estimated position
x_estimated = xhat_history_GN[:, -1]

plot_estimation_history(xhat_history_GN, J_history_GN)

# Convert headings to degrees for printing
x_true_deg = x_true.copy()
x_estimated_deg = x_estimated.copy()
x_true_deg[2] = np.degrees(x_true[2])
x_estimated_deg[2] = np.degrees(x_estimated[2])

# Print the results
print(f"True Position: x = {x_true_deg[0]:.2f} cm, y = {x_true_deg[1]:.2f} cm, heading = {x_true_deg[2]:.2f}°")
print(f"Estimated Position: x = {x_estimated_deg[0]:.2f} cm, y = {x_estimated_deg[1]:.2f} cm, heading = {x_estimated_deg[2]:.2f}°")

# Plot the results
plot_robot_position(x_true, x_estimated, qr_pos)