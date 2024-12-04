In task 2, the measurements/reading has been recorded in each direction/axis in the following order:
+z axis, -z axis, +x axis, -x axis, +y axis, -y axis.

Note, the columns in the measurement log file represents
timestamp,
accelerometer a_x (negative a_x),
accelerometer a_y (negative a_y),
accelerometer a_z (positive a_z),
roll,
pitch,
gyroscope g_x (negative g_x),
gyroscope g_y (negative g_y),
gyroscope g_z (positive g_z),
megnetometer mg_x (negative mg_x),
megnetometer mg_y (negative mg_y),
megnetometer mg_z (positive mg_z)

This is due to the body of the robot and camera has the following direction:
       z
 y     ^
 ^     |
  *    |
   *   |
    *  |
     * |------------------> x
     
while the IMU sensors have with respect to robot body and camera sensor

        z
 -y     ^
 ^      |
  *     |
   *    |
    *   |
     *  |
       *|------------------> (-x)
 


 
