# vive_ros2
Code for using the HTC vive tracking system with ROS2. This branch of the package is designed 
to be used without ROS. 

This package allows for maximum flexibility in terms of where the HTC drivers are run
since the package uses an independent server client architecture using python socket library.
For example, server can be run on a more powerful Windows machine for best driver support while
the client can be run on the robot or target device. The server and client can also
be run on the same device if desired. 

## Getting Started
### Running the Python Server
The best way to get started is to run `setup.bat` which will create a python virtual environment with 
the necessary dependencies. Make sure to have `SteamVR` installed and running with a tracker connected 
before starting the client or server. `SteamVR` must be configured as detailed in our internal documentation. 

Once you start the server you should see something like the following printed to the screen indicating which VR devices 
are detected and what the ip address and port (IP:PORT) of the server is. This information can be used when 
setting up the client.

```
13:49:17|ViveTrackerServer|INFO|Starting server at 192.168.50.171:8000
13:49:17|ViveTrackerServer|INFO|Connected VR devices: 
###########
Found 4 Tracking References
  tracking_reference_1 (LHB-59F8D726, Mode Valve SR Imp, Valve SR Imp)
  tracking_reference_2 (LHB-AA3BEC72, Mode Valve SR Imp, Valve SR Imp)
  tracking_reference_3 (LHB-FFCE0AD4, Mode Valve SR Imp, Valve SR Imp)
  tracking_reference_4 (LHB-91047ECC, Mode Valve SR Imp, Valve SR Imp)
Found 1 HMD
  hmd_1 (LHR-8280F84D, VIVE_Pro MV)
Found 1 Controller
  controller_1 (LHR-4F3DC6EA, VIVE Controller Pro MV)
Found 1 Tracker
  tracker_1 (LHR-55804C5D, VIVE Tracker Pro MV)
###########
```

The provided script `server_with_one_client.py` will start a client and server, enabling recording of data
when a single tracker is connected via `SteamVR`. 