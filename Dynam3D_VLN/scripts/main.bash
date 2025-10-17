export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TORCH_DISTRIBUTED_DEBUG=INFO

flag1="--exp_name release
      --run-type train
      --exp-config scripts/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 1
      IL.iters 100000
      IL.lr 1e-6
      IL.log_every 1000
      IL.ml_weight 1.0
      IL.sample_ratio 0.8
      IL.load_from_ckpt False
      IL.is_requeue False
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      "

flag2=" --exp_name release
      --run-type eval
      --exp-config scripts/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release/ckpt.iter8000.pth
      IL.back_algo control
      "

flag3="--exp_name release
      --run-type inference
      --exp-config scripts/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 4
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH data/logs/checkpoints/release/ckpt.iter12000.pth
      INFERENCE.PREDICTIONS_FILE preds.json
      IL.back_algo control
      "

mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -m torch.distributed.launch --nproc_per_node=4 --master_port $2 --use_env run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      CUDA_VISIBLE_DEVICES='0' python3 -m torch.distributed.launch --nproc_per_node=1 --master_port $2 --use_env run.py $flag2
      ;;
      infer)
      echo "###### infer mode ######"
      CUDA_VISIBLE_DEVICES='0' python3 -m torch.distributed.launch --nproc_per_node=1 --master_port $2 --use_env run.py $flag3
      ;;
esac