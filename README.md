# deep-sleep-pytorch
<!-- Dasdf;lkj -->
<!-- ========== -->

## Requirements
Principal requirements are Python 3.7.3, PyTorch 1.1.0, CUDA 10 and cuDNN 7.5.1.
An environment YAML has been provided with other packages required for this repository, which can be installed by running `conda env create -f env.yaml`.

## Preliminaries
In the following, `COHORT_NAME` will designate the name of a custom cohort.

## Data preparation

### Set up channel label JSON file
Use the `utils/channel_label_identifier.py` tool by running
```
python src/utils/channel_label_identifier.py <path to folder containing EDFs and hypnograms> src/configs/signal_labels/<COHORT_NAME>.json C3 C4 A1 A2 EOGL EOGR LChin RChin EMG
```
This will create a JSON file containing key-value pairs to map the desired electrode labels shown above with the electrode configurations available in the data.

### Setup data processing configuration file
1. Copy the contents of `src.configs.data_base.json` to another file `data_<COHORT_NAME>.json` in the same directory.
2. Insert the name of your test data `COHORT_NAME` in line 3 of the file, and change the `edf` and `stage` paths to point to the location of your EDFs and hypnograms (this can be the same directory).
3. (optional) Change the output directory to a custom location

### Run data pipeline to generate H5 files
1. Modify the code that returns a list of hypnograms (around line 320) for your specific use-case.
2. Add a routine to extract subject ID (`list_subjectID`) from filenames around line 363.
3. Add a routine to extract hypnogram in the `process_file()` function around line 118. The output shape of the `hypnogram` variable should be `(N,)` (a 1D array), where `N` is the number of 30 s epochs.
4. If you have lights-off/on information, you can include a routine in `process_file()` around line 266.
5. If you have non-AASM standard sleep scoring in your hypnograms (ie. values outside of {W, N1, N2, N3, R} --> {0, 1, 2, 3, 4}), you can add a routine around line 282.

Now run the data generation pipeline using
```
python -m src.data.generate_cohort_files -c data_<COHORT_NAME>.json
```
this will generate the H5 files containing the EDF/hypnogram data, and a CSV file containing an overview over the used files.

## Inference on new data
### Set up configuration file
1. Change the `data.test` parameter `config.yaml` corresponding to your `COHORT_NAME` variable.
2. Change the `exp.name` parameter to your `EXPERIMENT_NAME`.
3. Change the `trainer.log_dir` parameter to `experiments/<EXPERIMENT_NAME>`.
4. (Optional) If using more than 1 GPU, change the `trainer.n_gpu` parameter to the number of GPUs.
5. Change data.data_dir if output directory was set to custom location

### Run script
```
python predict.py -c config.yaml -r trained_models/best_weights.pth
```

## Citation
A. N. Olesen, P. J. Jennum, E. Mignot, H. B. D. Sorensen. Automatic sleep stage classification with deep residual networks in a mixed-cohort setting. *Sleep*, Volume 44, Issue 1, January 2021, zsaa161. [DOI:10.1093/sleep/zsaa161](https://doi.org/10.1093/sleep/zsaa161)
```
@article{Olesen2020,
    author = {Olesen, Alexander Neergaard and {J{\o}rgen Jennum}, Poul and Mignot, Emmanuel and Sorensen, Helge Bjarup Dissing},
    doi = {10.1093/sleep/zsaa161},
    journal = {Sleep},
    number = {1},
    pages = {zsaa161},
    title = {{Automatic sleep stage classification with deep residual networks in a mixed-cohort setting}},
    volume = {44},
    year = {2021}
}
```
