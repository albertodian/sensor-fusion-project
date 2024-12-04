For task 3, use the file camera_module_calibration_task3.csv
There are two columns in the csv file.
- First column is the distance between the QR code and the robot camera in cm. 
- Second column is the height of qr code detected by the camera in pixel.
In order to get the true distance between the QR code and the robot camera, you need to add the following distances:
- Distance of camera pinhole and the IR detector = 1.7 cm
- Distance of the wall and the wooden list/bar = 5cm

You can also check camera_reading_task3.csv to understand the data.
