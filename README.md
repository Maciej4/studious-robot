# Minecraft Agent

This project aims to create an LLM based agent to perform some simple tasks in Minecraft, using (in part) the pointing
capabilities of the Molmo VLM. This project is for the course Neural Computation (CSE 290D) at UCSC. The code is in need
of substantial refactoring and cleanup. In particular, the setup process is quite complex and several parts of the code
are fragile to external changes.

## How to run

This project requires a large amount of VRAM to be run. In particular, the Molmo VLM (even when quantized to 4bit, which
only works on CUDA for now) requires 12 GB of VRAM just to load the model and around 14 GB during inference.
Additionally, the reasoning LLM requires around 8 GB of VRAM during inference. This project has been tested with Molmo
running in a WSL environment with a 4080 Super GPU while the reasoning LLM runs a M1 Max Laptop with 32 GB of unified
memory (around 24 GB of effective VRAM). Flask APIs are used to communicate between the two environments.

1. Install and launch WSL (tested with Ubuntu 22.04.5 LTS).

Within WSL:

2. Create a python virtual environment and install the requirements in `requirements_wsl.txt`.
3. Run `python3 host_model.py` which will download and launch the Molmo VLM model as well as its API.

Now outside of WSL (on the host machine):

4. Outside of WSL, launch Minecraft and open a world.
5. Run `pip install opencv-python Flask PyAutoGUI` in a python venv to install the required packages.
6. Run `python3 controls.py` which will launch the API to control the agent in Minecraft.
    - You will need to download the invicons from the [Minecraft Wiki](https://www.minecraft.wiki). Go to the page of an
      item you want the model to be able to regonize and look to the right side of the page. There should be a large
      image of the item with a smaller image inside a gray square. Right click and save the smaller image. For
      reference, here is inventory icon for the wooden
      pickaxe: [Wooden Pickaxe](https://minecraft.wiki/images/Invicon_Wooden_Pickaxe.png?86864). Without renaming this
      file, move it to a new folder called images in the root of this project.
7. Click on the Minecraft window to give it focus (some Minecraft settings may need to be changed).
    - In the game's settings disable raw input.

For the machine with the reasoning LLM:

8. Run a OpenAI API compatible server with an LLM model. This project uses LMStudio to host the
   model: https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B.
9. Install the requirements in `requirements.txt` in a python venv.
10. Add a new file in the root directory called `env.json` containing a single entry `controls_base_url` with the value
    of the base URL of the controls API.
    - For example, if the controls API is running on `http://localhost:5000`, the `env.json` file should contain
      `{"controls_base_url": "http://localhost:5000"}`.
11. Finally, run `python3 simple_agent.py` to launch the agent.

### Acknowledgements

This project makes use of inventory icons taken from the [Minecraft Wiki](https://www.minecraft.wiki) for the purposes
of recognizing items in the inventory. This capability can probably be replaced by looking for new items in the
inventory, hovering over them, and reading the tooltip. However, this is beyond the current scope of the project.

