# Install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash ~/Anaconda3-2020.02-Linux-x86_64.sh -b -p
source ~/anaconda3/bin/activate

# Create python 3.7 env
conda create -n monetenv -q python=3.7 -y
conda activate monetenv

# Install pytorch and utils
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch -y

# Install required packages
sudo apt install ninja-build coinor-cbc coinor-libcbc-dev -y
conda config --add channels http://conda.anaconda.org/gurobi
conda install -c conda-forge cvxpy gurobi -y
conda install -c anaconda pandas -y
yes | pip install cylp
