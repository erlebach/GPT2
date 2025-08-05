metrics_11189751.csv
34 GB peak, 0.1 sec per step 
Memory allocated: 2.7 GB
----------------------------------------------------------------------
JOBID: 11189838: 1 GPU
JOBID: 11189840: 2 GPUs
Issue: DDP should lead to less memory per GPU on each processor since the 
batch data is sharded across GPUs. 
----------------------------------------------------------------------
JOBID: 11189846: 2 GPUs
- Print the value of the batch shape to confirm whether or not the batch is sharded
across GPUs. 
- Result: there was no sharding. 
----------------------------------------------------------------------
JOBID: 11189849
- modify dataloader to handle sharding (I must check docs and provide a link)
----------------------------------------------------------------------
JOBID: 11190300
- Run a code with fabric that no longer uses Trainer class from Lightning, but stil 
  uses the structure required by LightningModule
----------------------------------------------------------------------
JOBID: 11190348
- I added Sync barriers at the end of training to ensure that one of the GPUs does not exit 
  main() without both checking various conditions. Prints only occur for Rank-0 (GPU-0).
----------------------------------------------------------------------
JOBID: 11190378
- confusion due to timings of training_step in lightning module and fabric module. 
- this led to code cleanup
----------------------------------------------------------------------
jOBID: 11190632
- 2 GPUs
- batch_size=64
- print GPU diagnostics for 2 steps
- max_steps=20
----------------------------------------------------------------------
JOBID: 11190640
- 1 GPU
- batch_size=64
- max_steps = 20
- Times are slightly lower than with 2 GPUs. That makes sense since there is no communication. 
- One must distinguish inner training_step (lightning) and outer training__step (fabric). 
  the fabric step to take longer with 2 GPUs but the lightning training_step to be the same since
  there are no synchronization/communication that will slow things down.
----------------------------------------------------------------------
JOBID: 11190645
- 2 GPUs
- batch_size = 32
- max_steps=20
- Why are there two different times for training_step in lightning
----------------------------------------------------------------------
JOBID: 11190808
- batch_size=32
- two GPUs
Objective: try to understand whether batch size changes within an epoch. 
----------------------------------------------------------------------
JOBID: 11190824
- batch_size 32
- Single GPU
----------------------------------------------------------------------
JOBID: 11190865
- disable memory monitoring to see if timings are more stable. 
----------------------------------------------------------------------
JOBID: 11191019
- Added additional tests. See analysis in Understanding\ timing...md
- No memory measurements in lightning. I will now repeat the experiment with both timing and mmeory measurments. 
  Memory measurements in fabric. 
----------------------------------------------------------------------
JOBID: 11191029
- single GPU
- Memory disabled for al decorators (so I can compare apples and apples)
----------------------------------------------------------------------
JOBID: 
- single GPU
- Memory enabled for al decorators (so I can compare apples and apples)
- the timings are very consistent: 0.09 and 0.33 (there are no variations). All variations were due to 
  the measurement of memory. 
----------------------------------------------------------------------
JOBID: 11192212
- timing experiments: over batch size and model size. 
- Output saved in json rather than csv
- explicit loop timings rather than using decorators. 
- Output data in: timing_scaling_experiment_20250805_134903.json
----------------------------------------------------------------------
JOBID: 11192222
- Added a 3rd experiments: over sequences
- output saved in json
- measure memory usage across experiments
- Output data in: memory_scaling_experiment_20250805_140102.json
----------------------------------------------------------------------
JOBID: 11192518
- Output data in: memory_scaling_experiment_20250805_172941.json
- Separate out forward and backward modes. 
----------------------------------------------------------------------
JOBID: 
- Output data in: 
- Implement eval mode (doubles output). Peak memory in training mode will be 
  much higher than peak memory in evaluation mode. This serves as a check. 
----------------------------------------------------------------------
