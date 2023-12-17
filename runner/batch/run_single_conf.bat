@echo off

set "algos=qmix_cyborg"
set "wrappers=table"
set "runs=1"
set "map=availability_medium"

for %%A in (%wrappers%) do (
    for %%X in (%algos%) do (
        for /l %%Y in (1, 1, %runs%) do (
            echo Trial %%X_%map%_%%A: Iteration %%Y
            python "pymarl2\main.py" --config=%%X --env-config=cyborg with env_args.map_name=%map% env_args.wrapper_type=%%A t_max=1005000 name=%%X_%map%
        )
    )
)