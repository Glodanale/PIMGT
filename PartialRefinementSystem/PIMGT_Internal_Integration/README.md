## Train

- Check `requirements.txt`
- Train PIMGT_Internal:
  ```shell
  python main.py <dataset> PIMGT_Internal <experiment name> <CUDA device>
  ```
  For example, 
  ```shell
  python main.py I80 PIMGT_Internal E01 0
  ```
  means training PIMGT_Internal model for dataset I80, the experiment name is E01, and the CUDA device number is 0.
- The experiment results will be under the directory: `exps/I80/PIMGT_Internal/E01`

- Note: Update seed value in main.py main function at bottom of file