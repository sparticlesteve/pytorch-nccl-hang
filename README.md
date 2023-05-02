# Simple PyTorch reproducer of the NCCL SS11 hangs

This example was provided by Josh Romero from NVIDIA.

Submit the test with

```
sbatch -N 8 test_job.sh
```

The job will run 100 trials of the `run_test.sh` script, which will run the
`simple_pyt.py` training example in a data-parallel fashion with NCCL.
