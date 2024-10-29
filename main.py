import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset1/data/task1/imu_reading_task1.csv')

print(df.head())
print(df.info())

# Task 1.a - Visualize the data

df.columns = [
    'timestamp_ms',
    'ax_g', 'ay_g', 'az_g',
    'roll_deg', 'pitch_deg',
    'gx_deg_s', 'gy_deg_s', 'gz_deg_s',
    'mx_gauss', 'my_gauss', 'mz_gauss'
]

df['timestamp_s'] = df['timestamp_ms'] / 1000.0

plt.figure(figsize=(15, 8))

plt.plot(df['timestamp_s'], df['gx_deg_s'], label='Gyroscope X (deg/s)')
plt.plot(df['timestamp_s'], df['gy_deg_s'], label='Gyroscope Y (deg/s)')
plt.plot(df['timestamp_s'], df['gz_deg_s'], label='Gyroscope Z (deg/s)')

plt.title('Gyroscope Readings Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (deg/s)')
plt.legend()
plt.grid(True)
plt.show()

# Melt the DataFrame for easier plotting with seaborn
# gyro_data = df.melt(id_vars=['timestamp_s'], value_vars=['gx_deg_s', 'gy_deg_s', 'gz_deg_s'],
#                     var_name='Axis', value_name='Angular Velocity (deg/s)')

# plt.figure(figsize=(15, 6))
# sns.histplot(data=gyro_data, x='Angular Velocity (deg/s)', hue='Axis', kde=True, bins=50)
# plt.title('Distribution of Gyroscope Readings')
# plt.xlabel('Angular Velocity (deg/s)')
# plt.ylabel('Frequency')
# plt.show()

# Task 1.b - Determine the bias and variance of the gyroscope

# Calculate bias (mean) for each gyroscope axis
bias = {
    'gx_deg_s': df['gx_deg_s'].mean(),
    'gy_deg_s': df['gy_deg_s'].mean(),
    'gz_deg_s': df['gz_deg_s'].mean()
}

# Calculate variance for each gyroscope axis
variance = {
    'gx_deg_s': df['gx_deg_s'].var(),
    'gy_deg_s': df['gy_deg_s'].var(),
    'gz_deg_s': df['gz_deg_s'].var()
}

print("Gyroscope Bias (Mean):")
for axis, value in bias.items():
    print(f"{axis}: {value:.6f} deg/s")

print("\nGyroscope Variance:")
for axis, value in variance.items():
    print(f"{axis}: {value:.6f} (deg/s)^2")
