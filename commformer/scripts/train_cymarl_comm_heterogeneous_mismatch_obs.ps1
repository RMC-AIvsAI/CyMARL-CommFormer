# Trains 3 x CommFormer heterogeneous agents with two communication channels enabled each, in an identical network topology to that seen in CyMARL phase 2.
# Refer to the scenario file for specifics on agent actions and network topology.
python .\train\train_cymarl_comm.py --env_name "CybORG" --algorithm_name "commformer_dec" --scenario_name "phase2_confidentiality_small_heterogeneous_mismatch"`
--eval_episode_length 30 --action_limiting "True" --n_rollout_threads 48 --episode_length 30 --num_env_steps 80000000 --log_interval 10 --use_bilevel "True" --seed 4 --alg_seed 4 --use_eval "True" --eval_interval 10 --n_eval_rollout_threads 2 --eval_episodes 2 --sparsity 0.7

# Shut down the computer after the script completes
Start-Process -FilePath "shutdown.exe" -ArgumentList "/s /t 0"