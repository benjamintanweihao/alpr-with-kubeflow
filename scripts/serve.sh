echo $@
#1 project_workspace
#2 model_dns_prefix

python $1/serving/kfs_deployer.py --name=$2
