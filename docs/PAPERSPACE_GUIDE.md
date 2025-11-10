### Paperspace Guide

## Initialization Steps

1. Start machine in a gradient notebook
2. Mount DigitalOcean bucket in data sources
3. In terminal:
    A. Install required global packages
        - sudo apt install python3-venv
        - sudo apt install gh
        - sudo apt install screen
    B. Initialize virtual env
        - python3 -m venv .venv
        - source .venv/bin/activate
    C. Initialize git repo
        - sudo gh auth login
            - Authenticate via web browser/https
        - git config --global --edit
        - sudo git clone https://github.com/Marauder-Robotics/computer-vision-training.git

## Editing Codebase

1. Open Jupyterlab in DigitalOcean and make edits as needed
2. Push changes
    - In terminal:
        - git push origin branch-name
    - JupyterLab:
        - if logged in, will be able to use UI to push to branches
        - main protected, so cannot push directly to main (need to create PR in GitHub to add to main)
    
## Running files

1. Notes:
    - Paperspace gradient is built to run via .ipynb file. However, in this project it is 
      easiest to run from scripts via the terminal. All outputs are saved to file, so no
      need to view terminal output. If desired, can save terminal output to a screen file.
    - Terminal instances are the same in Jupyterlab and Paperspace, just differing UI's.
2. Run scripts
    - With screen:
        - screen -L -Logfile /notebooks/computer-vision-training/cv_training.log -S cv_training
        - chmod +x path_to_script
        - pip install -r requirements.txt
        - python path_to_file --vars
