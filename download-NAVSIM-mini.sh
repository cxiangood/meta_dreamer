mkdir -p navsim_mini && cd navsim_mini

BASE="https://hf-mirror.com/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1"

# 1) metadata
wget -c --tries=20 --waitretry=5 "${BASE}/openscene_metadata_mini.tgz"
tar -xzf openscene_metadata_mini.tgz
rm -f openscene_metadata_mini.tgz
mv openscene-v1.1/meta_datas mini_navsim_logs

# 2) camera blobs
for i in $(seq 0 31); do
  f="openscene_sensor_mini_camera_${i}.tgz"
  wget -c --tries=20 --waitretry=5 "${BASE}/openscene_sensor_mini_camera/${f}"
  tar -xzf "${f}"
  rm -f "${f}"
done

# 3) lidar blobs
for i in $(seq 0 31); do
  f="openscene_sensor_mini_lidar_${i}.tgz"
  wget -c --tries=20 --waitretry=5 "${BASE}/openscene_sensor_mini_lidar/${f}"
  tar -xzf "${f}"
  rm -f "${f}"
done

mv openscene-v1.1/sensor_blobs mini_sensor_blobs
rm -rf openscene-v1.1
echo "Done: $(pwd)"
