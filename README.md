# EECS 595 Natural Language Processing Final Project: Song Genre Classification based on Audio Metadata

These instructions will guide you to reproduce the results found in our paper. Note that the instructions are geared towards Mac users.

# Setup
1. Git clone this repository to a directory of your choice
2. `cd` to the directory where this project is stored
3. Create a virtual environment with `python3 -m venv env`
4. Active the virtual env with `source bin/activate`
5. Install required dependencies for the code to run with `pip install -r requirements.txt`

# Running the Code
The main file used to run the code is `read_file.py`. As you can see, we have already trained and saved models in the `*.torch` files, and we can use these models to reproduce the results found in our paper.

Before we get to actual commands to execute our code, we will introduce the command options available for usage. If some of these options do not make sense, please read the paper first to understand context.

`--train`: Option to train a model. Leaving this option out will evaluate a model instead.

`--gpu`: Option to utilize the GPU

`--seg_comp_mode [PV | WPV | GF | WGF]`: Option to specify segment compilation mode

`--sec_comp_mode [PV | WPV | GF | WGF]`: Option to specify section compilation mode

`--model_in <model_name>`: Option to specify a model to evaluate.

# Example Commands
Train a model in PV segment compilation mode utilizing the GPU:

`python read_file.py --train --gpu --seg_comp_mode PV`
<br/><br/>

Evaluate the `model_WGF_6.torch` model in PV segment compilation mode and WGF section compilation mode:

`python read_file.py --seg_comp_mode PV --sec_comp_mode WGF --model-in model_WGF_6.torch`