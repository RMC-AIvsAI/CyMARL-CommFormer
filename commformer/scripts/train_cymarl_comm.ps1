# Execute the Python script with the specified arguments
python .\train\train_cymarl_comm.py --env_name "CybORG" --algorithm_name "commformer_dec" --scenario_name "phase2_confidentiality_small_heterogeneous"`
--num_agents 2 --eval_episode_length 10 --action_limiting "True" --n_rollout_threads 32 --episode_length 30 --num_env_steps 1000000 --log_interval 10 --use_bilevel "True" --seed 4