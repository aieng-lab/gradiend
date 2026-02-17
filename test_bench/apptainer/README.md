# Apptainer Configuration for Test Bench

This directory contains the Apptainer definition file for building a container image for running the test bench locally.

**Single-call (non-SLURM):** From repo root, run `./test_bench/run_test_bench.sh --apptainer` to build the SIF (if needed) and run the full test bench with fresh cache and results overview.

## Building the Image

### On Login Node

```bash
cd /scratch/$USER/gradiend
apptainer build /scratch/$USER/apptainer/images/gradiend-test-bench.sif \
    test_bench/apptainer/gradiend-test-bench.def
```

### What's Included

- Base: `continuumio/miniconda3:latest`
- Environment files copied into image
- System dependencies (git, etc.)
- Base conda environment (will be overridden by user's env in /scratch)

## Image Structure

The image is minimal - it just provides the base conda installation. The actual conda environment is created in `/scratch/$USER/conda-envs/gradiend` on first run, which:
- Persists across jobs
- Can be updated without rebuilding the image
- Is specific to each user

## Updating the Image

If you need to update the base image or add system dependencies:

1. Edit `gradiend-test-bench.def`
2. Rebuild: `apptainer build <output>.sif gradiend-test-bench.def`
3. The conda environment in `/scratch` will be reused (no need to recreate)

## Notes

- The image is relatively small (~500MB) since it's just base Miniconda
- Environment files are copied at build time for reference
- Actual environment creation happens at runtime in `/scratch`
