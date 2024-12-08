import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load IMU data
imu_data_file = 'dataset1_part2/data/task6-task7/imu_tracking_task6.csv'
imu_df = pd.read_csv(imu_data_file, header=None)
imu_df.columns = [
    'timestamp_ms',
    'ax_g', 'ay_g', 'az_g',
    'roll_deg', 'pitch_deg',
    'gx_deg_s', 'gy_deg_s', 'gz_deg_s',
    'mx_gauss', 'my_gauss', 'mz_gauss'
]

# Preprocess IMU data
skip_measurements = 1000
imu_df = imu_df[skip_measurements:]  # Remove first 1000 samples
imu_df['timestamp_s'] = imu_df['timestamp_ms'] # Convert to seconds
imu_df['ax_m_s2'] = imu_df['ax_g'] * 9.8
imu_df['ay_m_s2'] = imu_df['ay_g'] * 9.8
imu_df['gz_rad_s'] = np.deg2rad(imu_df['gz_deg_s'])

# Load Camera data
camera_file = 'dataset1_part2/data/task6-task7/camera_tracking_task6.csv'
camera_df = pd.read_csv(camera_file, header=None, index_col=False)
camera_df.columns = ['timestamp_ms', 'qr_code', 'c_x', 'c_y', 'width', 'height', 'distance_cm', 'attitude_deg']
camera_df = camera_df[skip_measurements:]  # Remove first 1000 samples
camera_df['timestamp_s'] = camera_df['timestamp_ms']  # Convert to seconds
camera_df['distance_m'] = camera_df['distance_cm'] / 100.0
camera_df['bearing_rad'] = np.deg2rad(camera_df['attitude_deg'])

# Load QR code positions
qr_code_file = 'dataset1_part2/data/qr_code_position_in_global_coordinate.csv'
qr_df = pd.read_csv(qr_code_file)
qr_df.columns = ['qr_code', 'mid_point_x_cm', 'mid_point_y_cm', 'position_in_wall']

# Create a dictionary for QR code positions
qr_positions = {}
for idx, row in qr_df.iterrows():
    qr_id = row['qr_code']
    qr_positions[qr_id] = (row['mid_point_x_cm'] / 100.0, row['mid_point_y_cm'] / 100.0)

# Synchronize IMU and Camera data
start_time = imu_df['timestamp_s'].min()
end_time = imu_df['timestamp_s'].max()
total_duration = end_time - start_time 
print(f"Total recording duration: {total_duration:.2f} seconds")

# Interpolation for IMU data
imu_time = imu_df['timestamp_s'].values
ax_interpolator = interp1d(imu_time, imu_df['ax_m_s2'].values, kind='linear', fill_value="extrapolate")
gz_interpolator = interp1d(imu_time, imu_df['gz_rad_s'].values, kind='linear', fill_value="extrapolate")

# Prepare to store synchronized measurements
num_steps = int((end_time - start_time) / 0.1) + 1  # Number of steps based on 10ms intervals
camera_measurements = [[] for _ in range(num_steps)]  # List of lists to hold multiple measurements

# Align Camera Data
time_stamps = np.arange(start_time, end_time, 0.1)  # 10ms time step

for idx, row in camera_df.iterrows():
    camera_time = row['timestamp_s']

    # Interpolate IMU data at the camera timestamp
    ax_value = ax_interpolator(camera_time)
    gz_value = gz_interpolator(camera_time)

    qr_id = int(row['qr_code'])
    if qr_id in qr_positions:
        z_k = np.array([row['distance_m'], np.deg2rad(row['attitude_deg'])])
        index = int((camera_time - start_time) / 0.1)
        if 0 <= index < len(camera_measurements):
            camera_measurements[index].append({'qr_id': qr_id, 'z_k': z_k})

# 5. Define EKF Functions
def process_model(x, u, dt):
    x_pos, y_pos, theta, v, omega = x
    a_k, alpha_k = u
    x_pos += v * np.cos(theta) * dt
    y_pos += v * np.sin(theta) * dt
    theta += omega * dt
    v += a_k * dt
    omega += alpha_k * dt
    return np.array([x_pos, y_pos, theta, v, omega])

def process_model_jacobian(x, u, dt):
    x_pos, y_pos, theta, v, omega = x
    a_k, alpha_k = u
    F = np.eye(5)
    F[0, 2] = -v * np.sin(theta) * dt
    F[0, 3] = np.cos(theta) * dt
    F[1, 2] = v * np.cos(theta) * dt
    F[1, 3] = np.sin(theta) * dt
    F[2, 4] = dt
    return F

def measurement_model(x, qr_position):
    x_pos, y_pos, theta, _, _ = x
    x_qr, y_qr = qr_position
    dx = x_qr - x_pos
    dy = y_qr - y_pos
    r = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx) - theta
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    return np.array([r, phi])

def measurement_jacobian(x, qr_position):
    x_pos, y_pos, theta, _, _ = x
    x_qr, y_qr = qr_position
    dx = x_qr - x_pos
    dy = y_qr - y_pos
    r_squared = dx**2 + dy**2
    r = np.sqrt(r_squared)
    H = np.zeros((2, 5))
    H[0, 0] = -dx / r
    H[0, 1] = -dy / r
    H[1, 0] = dy / r_squared
    H[1, 1] = -dx / r_squared
    H[1, 2] = -1
    return H

# Initialize state and covariance
initial_X = 62.7 / 100.0
initial_Y = 16.0 / 100.0
x_est = np.array([initial_X, initial_Y, 0.0, 0.0, 0.0])
P_est = np.diag([1.0, 1.0, np.radians(5), 1.0, np.radians(5)])**2

Q = np.diag([0.3, 0.1, np.radians(1), 0.5, np.radians(0.2)])**2
R = np.diag([0.1, np.radians(5)])**2

# 7. Run the EKF
x_estimates = np.zeros((num_steps, 5))
x_estimates[0] = x_est

for k in range(1, num_steps):
    dt_k = 0.1  # 10ms time step
    a_k = ax_interpolator(time_stamps[k])
    omega_k = gz_interpolator(time_stamps[k])
    u_k = np.array([a_k, 0.0])  # Assuming alpha_k = 0

    # Prediction Step
    x_pred = process_model(x_est, u_k, dt_k)
    F_k = process_model_jacobian(x_est, u_k, dt_k)
    P_pred = F_k @ P_est @ F_k.T + Q

    # Update Step
    if camera_measurements[k]:
        for meas in camera_measurements[k]:
            qr_id = meas['qr_id']
            z_k = meas['z_k']
            qr_position = qr_positions[qr_id]
            z_hat = measurement_model(x_pred, qr_position)
            y_k = z_k - z_hat
            y_k[1] = (y_k[1] + np.pi) % (2 * np.pi) - np.pi
            H_k = measurement_jacobian(x_pred, qr_position)
            S_k = H_k @ P_pred @ H_k.T + R
            K_k = P_pred @ H_k.T @ np.linalg.inv(S_k)
            x_pred = x_pred + K_k @ y_k
            P_pred = (np.eye(5) - K_k @ H_k) @ P_pred

    # Update estimates
    x_est = x_pred
    P_est = P_pred
    x_estimates[k] = x_est

# 8. Visualize Results
x_positions = x_estimates[:, 0] 
y_positions = x_estimates[:, 1] 

print('Estimated Final Position:', x_positions[-1], y_positions[-1])

plt.figure(figsize=(10, 8))
plt.plot(x_positions, y_positions, label='Estimated Trajectory')
qr_x = [pos[0] for pos in qr_positions.values()]
qr_y = [pos[1] for pos in qr_positions.values()]
plt.scatter(qr_x, qr_y, marker='x', color='red', label='QR Code Positions')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Robot Trajectory Estimated by EKF')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()

# Plot heading angle
theta_estimates = x_estimates[:, 2]
time_axis = time_stamps

plt.figure(figsize=(10, 4))
plt.plot(time_axis, np.rad2deg(theta_estimates), label='Estimated Heading Angle')
plt.xlabel('Time (s)')
plt.ylabel('Heading Angle (degrees)')
plt.title('Estimated Heading Angle Over Time')
plt.legend()
plt.grid()
plt.show()