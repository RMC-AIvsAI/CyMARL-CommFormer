@echo off

for /F "tokens=1,2,3 delims=,*" %%A in (.\runner\config\baseline_models.txt) do (
            echo Eval %%A_%%B_%%C
            python "pymarl2\main.py" --config=%%A --env-config=cyborg with env_args.map_name=%%B env_args.wrapper_type=table t_max=1000 evaluate=True checkpoint_path=%%C name=eval_baseline_%%A_%%B
)