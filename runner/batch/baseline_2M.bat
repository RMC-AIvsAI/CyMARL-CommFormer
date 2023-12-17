@echo off

set "algos=iql_cyborg qmix_cyborg"
set "runs=5"

for /F "tokens=1,2 delims=,*" %%A in (.\runner\config\misinform_maps.txt) do (
    for %%X in (%algos%) do (
        for /l %%Y in (1, 1, %runs%) do (
            echo Trial %%X_%%A: Iteration %%Y
            python "pymarl2\main.py" --config=%%X --env-config=cyborg with env_args.map_name=%%A env_args.wrapper_type=table t_max=2005000 name=misinform_%%X_%%A
        )
    )
)