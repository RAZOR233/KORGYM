
## 🔔 Introduction



**KORGym** comprises over fifty games across six reasoning dimensions: mathematical and logical reasoning, control interaction reasoning, puzzle reasoning, spatial and geometric reasoning, strategic reasoning, and multimodal reasoning. The platform is structured into four modular components—the Inference Module, the Game Interaction Module, the Evaluation Module, and the Communication Module—which collectively enable multi-round evaluations, configurable difficulty levels, and stable reinforcement-learning support.

---

## ⚙️ Installation 

To install the required packages, run:

```bash
# Prepare repository and environment
git clone 'url'
cd ./KORGym
pip install -r requirements.txt
```
## 🛠️ eval_lib

### scritps
Please run the run.sh script in this folder to quickly launch KORGym.


### eval.py
This file primarily handles the overall evaluation process, including argument parsing, protocol setup, game environment initialization, and LLM-environment interaction.


### eval_lib.py
This file primarily handles the overall evaluation process, including argument parsing, protocol setup, game environment initialization, and LLM-environment interaction.


### utils.py
This file provides utility functions for argument parsing, format validation, and other helper tasks.


## 🗒️ Results
This folder contains all of KORGym’s model inference results, prompts, experimental test data, and intermediate game state data.


## 🎮 game_lib
This folder contains all the game files used in KORGym.



