wsl -d Ubuntu-24.04
su - clj

cd ~/smart_driving/Prj_worldmoudle
source dreamerv3/bin/activate
cd ~/smart_driving/Prj_worldmoudle/dreamerv3-main/dreamerv3-main
# python3 
python3 dreamer/dreamerv3/main.py --configs metadrive_lane_keeping --logdir ./logs
