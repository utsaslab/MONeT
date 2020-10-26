This folder contains the code for our implementation of [Checkmate](https://github.com/parasj/checkmate) in PyTorch, used for comparison with MONeT.

`checkmate_solver.py` has been copied almost as-in and is under Apache-2.0 license.

`checkmate_schedule.py` runs a Checkmate schedule.

They can be run using:

```
python checkmate_solver.py [MODEL] [BATCH_SIZE] [BUDGET]
python checkmate_schedule.py [MODEL] [BATCH_SIZE] [BUDGET] normal GUROBI --solution_file [PATH_TO_SOLUTION_FILE] --check_runtime
```
