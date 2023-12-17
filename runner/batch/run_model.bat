@echo off

set "algo=qmix"
set "wrapper=table"
set "runs=1"
set "map=confidentiality_medium"

set "model=C:\Users\wiebe\workspace\CyMARL\results\models\qmix_confidentiality_medium_table__2023-05-17_09-08-50"
set "step=1000560"

for /l %%Y in (1, 1, %runs%) do (
        echo Trial %%X_%map%_%%A: Iteration %%Y
        echo python "pymarl2\main.py" --config=%algo% --env-config=cyborg with env_args.map_name=%map% env_args.wrapper_type=%wrapper% t_max=2005000 name=%algo%_%map%_%wrapper% checkpoint_path=%model%
        python "pymarl2\main.py" --config=%algo% --env-config=cyborg with env_args.map_name=%map% env_args.wrapper_type=%wrapper% t_max=2005000 name=%algo%_%map%_%wrapper% checkpoint_path=%model% load_step=%step%
)
