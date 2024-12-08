import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

motor_control_file = 'dataset1_part2/data/task6-task7/motor_control_tracking_task6.csv'
motor_df = pd.read_csv(motor_control_file, header=None, names=['timestamp', 'left_pwm', 'right_pwm'])

motor_times = motor_df['timestamp'].values      # Timestamps
left_pwm = motor_df['left_pwm'].values             # Left PWM values
right_pwm = motor_df['right_pwm'].values           # Right PWM values

imu_data_file = 'dataset1_part2/data/task6-task7/imu_tracking_task6.csv'
columns = [
    'timestamp_ms',
    'ax_g', 'ay_g', 'az_g',
    'roll_deg', 'pitch_deg',
    'gx_deg_s', 'gy_deg_s', 'gz_deg_s',
    'mx_gauss', 'my_gauss', 'mz_gauss'
]
imu_df = pd.read_csv(imu_data_file, header=None, names=columns)

imu_times = imu_df['timestamp_ms'].values    # Timestamps in milliseconds
gyro_data_deg = imu_df['gz_deg_s'].values           # Gyroscope z-axis data in degrees/sec

start_index = 100
gyro_bias = np.mean(gyro_data_deg[:start_index])
gyro_data_deg_corrected = gyro_data_deg - gyro_bias

imu_times = imu_times[start_index:]
gyro_data_deg_corrected = gyro_data_deg_corrected[start_index:]

total_duration = imu_times[-1] - imu_times[0]
print(f"Total recording duration: {total_duration:.3f} seconds")

gyro_data_rad = gyro_data_deg_corrected * (np.pi / 180.0)

def zero_order_hold(motor_times, pwm_values, imu_times):
    pwm_interp = np.zeros_like(imu_times)
    idx = 0
    n_motor = len(motor_times)
    for i, t in enumerate(imu_times):
        while idx + 1 < n_motor and motor_times[idx + 1] <= t:
            idx += 1
        pwm_interp[i] = pwm_values[idx]
    return pwm_interp

left_pwm_interp = zero_order_hold(motor_times, left_pwm, imu_times)
right_pwm_interp = zero_order_hold(motor_times, right_pwm, imu_times)

plt.figure(figsize=(10, 5))
plt.plot(imu_times, left_pwm_interp, label='Left PWM')
plt.plot(imu_times, right_pwm_interp, label='Right PWM')
plt.xlabel('Time (s)')
plt.ylabel('PWM Value')
plt.title('Interpolated PWM Values Over Time')
plt.legend()
plt.grid(True)
plt.show()

k = 0.333
pwm_avg = (left_pwm_interp + right_pwm_interp) / 2.0

plt.figure(figsize=(10, 5))
plt.plot(imu_times, pwm_avg, label='Left PWM')
plt.xlabel('Time (s)')
plt.ylabel('PWM Value')
plt.title('Interpolated PWM Values Over Time')
plt.legend()
plt.grid(True)
plt.show()

v = k * pwm_avg    # Linear speed in m/s

plt.figure(figsize=(10, 5))
plt.plot(imu_times, v, label='Left PWM')
plt.xlabel('Time (s)')
plt.ylabel('PWM Value')
plt.title('Interpolated PWM Values Over Time')
plt.legend()
plt.grid(True)
plt.show()

px = [62.7 / 100.0]    # Initial X position in meters
py = [16.0 / 100.0]    # Initial Y position in meters
phi = [0.0]            # Initial heading angle in radians

delta_times = np.diff(imu_times, prepend=imu_times[0])

# Dead-reckoning loop
for i in range(1, len(imu_times)):
    delta_t = delta_times[i]
    v_i = v[i]
    omega_i = gyro_data_rad[i]

    # Update heading angle
    phi_new = phi[-1] + omega_i * delta_t
    # Normalize the angle to be within [-pi, pi]
    phi_new = (phi_new + np.pi) % (2 * np.pi) - np.pi
    phi.append(phi_new)

    # Update positions using the updated heading angle
    px_new = px[-1] + v_i * np.cos(phi_new) * delta_t
    py_new = py[-1] + v_i * np.sin(phi_new) * delta_t
    px.append(px_new)
    py.append(py_new)


px_cm = np.array(px) * 100.0
py_cm = np.array(py) * 100.0
phi_deg = np.array(phi) * (180.0 / np.pi) 

print('Estimated Final Position:', py_cm)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(px_cm, py_cm, label='Estimated Trajectory')
plt.xlabel('X Position (cm)')
plt.ylabel('Y Position (cm)')
plt.title('Robot Trajectory via Dead-Reckoning')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.plot(imu_times - imu_times[0], phi_deg, label='Heading Angle')
plt.xlabel('Time (s)')
plt.ylabel('Heading Angle (degrees)')
plt.title('Heading Angle over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()