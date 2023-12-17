@echo off

set "algos=iql qmix"
set "wrapper=table"
set "runs=3"
set "map=confidentiality_medium"

for %%X in (%algos%) do (
    for /F "tokens=*" %%A in (.\runner\config\hyperparam_batch_size.txt) do (
        for /F "tokens=*" %%B in (.\runner\config\hyperparam_buffer_size.txt) do (
            for /F "tokens=*" %%C in (.\runner\config\hyperparam_lr.txt) do (
                for /F "tokens=*" %%D in (.\runner\config\hyperparam_td_lambda.txt) do (
                    for /l %%Y in (1, 1, %runs%) do (
                        echo Trial %%X_%map%_%%A: Iteration %%Y
                        echo python "pymarl2\main.py" --config=%%X --env-config=cyborg with env_args.map_name=%map% env_args.wrapper_type=%wrapper% t_max=550000 name=%%X_%map%_hyperparam_%%A_%%B_%%C_%%D
                        python "pymarl2\main.py" --config=%%X --env-config=cyborg with env_args.map_name=%map% env_args.wrapper_type=%wrapper% t_max=550000 name=%%X_hyperparam_%%A_%%B_%%C_%%D %%A %%B %%C %%D
                    )
                )
            )
        )
    )
)