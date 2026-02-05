1st terminal:

source ~/.bashrc
cd ~/ardu_ws/
source ~/ardu_ws/install/setup.bash
ros2 launch ardupilot_gz_bringup multi_iris_jackal4.launch.py

----------------------------------------------------------
----------------------------------------------------------
2rd terminal:

source ~/.bashrc
cd ~/ardu_ws/
source ~/ardu_ws/install/setup.bash
mavproxy.py --console --map --aircraft test --master=:14550 --out 127.0.0.1:14551
----------------------------------------------------------
----------------------------------------------------------

3rd terminal

source ~/.bashrc
cd ~/ardu_ws/
source ~/ardu_ws/install/setup.bash
ros2 launch ardupilot_gz_bringup complete_control_system_v5.launch.py


----------------------------------------------------------
----------------------------------------------------------

clear all ports
sudo kill -9 $(sudo lsof -t -iUDP -iTCP)
killall -9 gz sim gz server ruby python3
ros2 daemon stop && ros2 daemon start

----------------------------------------------------------
----------------------------------------------------------
Important Note:
1) When we make changes in python file, and do colcon build on ~\ardu_ws\ the changes we made in python code does not gets executed, becuase the ros is executing a copy of the code installed in:
~/ardu_ws/install/ardupilot_gz_bringup/lib/ardupilot_gz_bringup$ 

we must delete that copy first and then do colcon build so that the chnages we have made apperas at the output.

2) Similarly when we make any changes in launch file, we have to delete the previous launch file availabel in installed folder so that chnages may take effect:
/home/bilal/ardu_ws/install/ardupilot_gz_bringup/share/ardupilot_gz_bringup/launch/multi_iris_jackal4.launch.py
