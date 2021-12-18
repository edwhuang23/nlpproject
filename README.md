# EECS 595 Natural Language Processing Final Project: Song Genre Classification from Pitch Data

These instructions will guide you to reproduce the results found in our paper. Note that the instructions are geared towards Mac users. Also note that the datasets we used are already included in the repo. If you would like to take a look at the original datasets, the Million Song Dataset can be found [here](http://millionsongdataset.com/), and the MSD Allmusic Genre Dataset can be found [here](http://www.ifs.tuwien.ac.at/mir/msd/partitions/msd-MAGD-genreAssignment.cls). In addition, there are a few files that are too big to be pushed onto this GitHub repository. Those files are accessible in [this](https://drive.google.com/drive/u/0/folders/0AJ1vx5o3L_LpUk9PVA) Google Drive folder. You can use [gdown](https://github.com/wkentaro/gdown) to download them to your directory or you can download directly from the Google Drive.

# Setup
1. Git clone this repository to a directory of your choice
2. `cd` to the directory where this project is stored
3. Create a virtual environment with `python3 -m venv env`
4. Active the virtual env with `source bin/activate`
5. Install required dependencies for the code to run with `pip install -r requirements.txt`

# Running the Code
The main file used to run the code is `dist_comp_segment.py`, `dist_comp_section.py`, or `dist_comp_song.py` depending on what distribution you would like to train on. As you can see, we have already trained and saved models in the `*.torch` files, and we can use these models to reproduce the results found in our paper.

Before we get to actual commands to execute our code, we will introduce the command options available for usage. If some of these options do not make sense, please read the paper first to understand context.

`--train`: Option to train a model. Leaving this option out will evaluate a model instead.

`--gpu`: Option to utilize the GPU

`--seg_comp_mode [PV | WPV | GF | WGF]`: Option to specify segment compilation mode in training.

`--sec_comp_mode [PV | WPV | GF | WGF]`: Option to specify section compilation mode in training.

`--model_in <model_name>`: Option to specify a model to evaluate. This option must be used for evluation.

# Example Commands
Train a model in PV segment compilation mode utilizing the GPU:

`python dist_comp_section.py --train --gpu --seg_comp_mode PV`
<br/><br/>

Evaluate the `model_PV_2e.torch` model in PV segment compilation mode and WGF section compilation mode:

`python dist_comp_section.py --seg_comp_mode PV --sec_comp_mode WGF --model-in model_PV_2e.torch`
