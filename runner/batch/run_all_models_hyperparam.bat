@echo off

for /F "tokens=1,2,3,4,5,6 delims=,*" %%A in (.\runner\config\hyperparam_models.txt) do (
            echo Eval %%A_%%B_%%C_%%D_%%E
            python "pymarl2\main.py" --config=%%A --env-config=cyborg with env_args.map_name=confidentiality_medium env_args.wrapper_type=table t_max=1000 evaluate=True checkpoint_path=%%F name=eval_%%A_%%B_%%C_%%D_%%E %%B %%C %%D %%E 
)