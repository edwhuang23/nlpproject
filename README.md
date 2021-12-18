# EECS 595 Natural Language Processing Final Project: Song Genre Classification from Pitch Data

These instructions will guide you to reproduce the results found in our paper. Also note that the datasets we used are already included in the repo. If you would like to take a look at the original datasets, the Million Song Dataset can be found [here](http://millionsongdataset.com/), and the MSD Allmusic Genre Dataset can be found [here](http://www.ifs.tuwien.ac.at/mir/msd/partitions/msd-MAGD-genreAssignment.cls). In addition, there are a few files that are too big to be pushed onto this GitHub repository. Those files are [dataset_limited_pr.tar.gz](https://drive.google.com/file/d/1-xD75AZfBBgdpKoLVNz9Rhyg5zXvlJ14/view?usp=sharing), [training_mod-4.0_limited_pr.pickle](https://drive.google.com/file/d/1P488dLQ_K9NdbFgpsJjapSkB7JPoZOzB/view?usp=sharing), [dataset.tar.gz](https://drive.google.com/file/d/1tcqHnYqD5_hUCpTkVjuzRS-hH1hsK6LO/view?usp=sharing), [training_mod-4.0.pickle](https://drive.google.com/file/d/1D04Gbi7BcmZWG29942z5RohlrtdIgRW0/view?usp=sharing), and [testing_mod-4.0.pickle](https://drive.google.com/file/d/1zLg19GpGlsYkHSzdoNAhkufRCDoEy-d5/view?usp=sharing). You can use [gdown](https://github.com/wkentaro/gdown) to download them to your directory or you can download directly from the Google Drive. The `testing_mod-4.0_limited_pr` pre-processed testing data for the limited_pr dataset is not provided, but `X_test_limited_pr` can be used in its place as it contains that same info, but all segments have been mapped to their corresponding index in the limited_pr vocabulary already.

# Setup
1. Git clone this repository to a directory of your choice
2. `cd` to the directory where this project is stored
3. Create a virtual environment with `python3 -m venv env`
4. For Mac (or if you are running this on GreatLakes), activate the virtual env with `source bin/activate`. For Linux, activate with `env\Scripts\activate`.
5. Install required dependencies for the code to run with `pip install -r requirements.txt`

# Running the Code
The main file used to run the code is `read_file.py`. As you can see, we have already trained and saved models in the `*.torch` files, and we can use these models to reproduce the results found in our paper.

Before we get to actual commands to execute our code, we will introduce the command options available for usage. If some of these options do not make sense, please read the paper first to understand context.

`--train`: Option to train a model. Leaving this option out will evaluate a model instead when running any of `dist_comp_models_*.py` or `vanilla_keras.py`.

`--gpu`: Option to utilize the GPU. This is an optional argument.

`--seg_comp_mode [PV | WPV | GF | WGF]`: Option to specify segment compilation mode. This option is required when evaluating any `*.torch` model with `dist_comp_models_*.py`, and it is required for training with `dist_comp_models_section.py` and `dist_comp_models_song.py`.

`--sec_comp_mode [PV | WPV | GF | WGF]`: Option to specify section compilation mode. This option is required when evaluating any `*.torch` model with `dist_comp_models_*.py`, and it is required for training with `dist_comp_models_song.py`.

`--model_in <model_name>`: Option to specify a model to evaluate. This option is required when evaluating any `*.torch` model with `dist_comp_models_*.py`. Do not include this option when evaluating with `vanilla_keras.py`.

# Example Commands
Train a model in PV segment compilation mode utilizing the GPU:

`python dist_comp_models_section.py --train --gpu --seg_comp_mode PV`
<br/><br/>

Evaluate the `model_WGF_6.torch` model in PV segment compilation mode and WGF section compilation mode:

`python dist_comp_models_section.py --seg_comp_mode PV --sec_comp_mode WGF --model-in model_WGF_6.torch`
<br/><br/>

To train a Keras model, `cd` into the `keras` directory and run

`python vanilla_keras.py --train`
