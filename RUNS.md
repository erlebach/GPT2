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
JOBID: 11190348.err
- I added Sync barriers at the end of training to ensure that one of the GPUs does not exit 
  main() without both checking various conditions. Prints only occur for Rank-0 (GPU-0).
----------------------------------------------------------------------
