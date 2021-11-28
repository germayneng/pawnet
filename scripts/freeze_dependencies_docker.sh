conda env export --no-builds > environment_docker.yml
# remove incompatible libraries for linux/ubuntu
gsed -i "/.*-\ libgfortran/d" environment_docker.yml
gsed -i "/.*-\ appnope/d" environment_docker.yml
echo "New docker environment.yml created."