# Sample Split Data

## Overview

The file `sample_split_data.npy` is a processed dataset that has been divided and ready for testing our program. It is processed using the script `extract_de.py`.

## Data Structure

### Features

The feature data in the dataset consists of several dimensions, defined as follows:

- `numSubject`: Number of subjects (individuals) in the data.
- `channels`: Number of channels used for recording the signals.
- `numTime`: Number of time points in the time series data. numTime = (signal_length - noverlap) // (nfft - noverlap)
- `numBand`: Number of frequency bands in the spectral data.

### Labels

The label data in the dataset is also organized in dimensions:

- `numSubject`: Number of subjects (individuals) in the data.
- `numTime`: Number of time points corresponding to the labels.

## Training and Testing Sets

The dataset is divided into training and testing subsets:

- **Training Set**: Contains data for two subjects.
- **Testing Set**: Contains data for one subject.

## Usage

To obtain the latent embedding, you can directly run the following command:

```bash
python main.py train config.yaml

