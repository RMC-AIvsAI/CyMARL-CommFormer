# Navigate to the directory containing the script
cd "c:\Users\user\.conda\envs\CyMARL-CommFormer\CyMARL-CommFormer\commformer\scripts\train"

# Execute the Python script with the specified arguments
python .\train_cymarl_comm.py --env_name "CybORG" --algorithm_name "commformer_dec" --scenario_name "confidentiality_small"`
--num_agents 2 --eval_episode_length 10 --n_rollout_threads 4 --seed 4 --episode_length 30 --num_env_steps 1000000