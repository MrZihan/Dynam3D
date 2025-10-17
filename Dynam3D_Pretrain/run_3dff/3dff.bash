export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export TORCH_DISTRIBUTED_DEBUG=DETAIL

flag1="--exp_name release_3dff
      --run-type train
      --exp-config run_3dff/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 1
      IL.iters 100000
      IL.lr 1e-5
      IL.log_every 1000
      IL.ml_weight 1.0
      IL.load_from_ckpt False
      IL.is_requeue False
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      "
#IL.ckpt_to_load data/logs/checkpoints/release_3dff/ckpt.iter100000.pth

flag2=" --exp_name release_3dff
      --run-type eval
      --exp-config run_3dff/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_3dff/ckpt.iter100000.pth
      "

mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      CUDA_VISIBLE_DEVICES='0' python3 -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag2
      ;;
esac