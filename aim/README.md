PICK and PLACE demo
=====================

Using Soft Actor Critic(SAC) to do pick and place 

Requirement:

-gym (rl interface)
-metaworld (environment)
-mujoco-py (using mujoco simulation)
-garage (algorithm)

Check requirement.txt for detail



### Run training

python sac_pick_and_place.py

### For demo 

If you train on server, you would need a GUI Ubuntu computer for demo

If GUI Computer has GPU and Cuda >=10.0:

python sim_policy_torch.py <your_snap_shot_file>

If GUI Computer doesnot have GPU and CUDA >= 10.0:

1. Convert model,env to cpu if you have trained on GPU before => it will output model_final.pt and env.pt 

python convert_gpu_cpu.py <your_snap_shot_file>

2. Run demo

python sim_policy_torch_v1.py model_final.pt env.pt




