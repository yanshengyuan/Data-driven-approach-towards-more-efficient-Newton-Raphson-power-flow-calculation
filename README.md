# ICT with Industry Workshop
Experimental repo for Newton Raphson initial value problem

## Getting Started

### Installation of Python

#### Windows
1. Download the latest version of Python from the [official website](https://www.python.org/downloads/).
2. Run the installer and ensure you check the box "Add Python to PATH".
3. Follow the installation steps.

#### macOS
1. Download the latest version of Python from the [official website](https://www.python.org/downloads/).
2. Open the downloaded package and follow the installation steps.

#### Linux
1. Open a terminal.
2. Install Python using your package manager. For example, on Debian-based systems:
    ```sh
    sudo apt update
    sudo apt install python3
    ```

### Creating a Virtual Environment

1. Open a terminal or command prompt.
2. Navigate to your project directory:
    ```sh
    cd /path/to/your/project
    ```
3. Create a virtual environment using the `venv` module:
    ```sh
    python3 -m venv .venv
    ```

### Activating the Virtual Environment

#### Windows
```sh
.\.venv\Scripts\activate
```

#### macOS and Linux
```sh
source .venv/bin/activate
```

### Installing Packages

1. Ensure your virtual environment is activated.
2. Install the required packages from `requirements.txt`:
    ```sh
    pip install -r requirements.txt
    ```

### Included Notebooks

#### pandapower_init_and_internals.ipynb
This notebook provides an introduction to the `pandapower` library, including its initialization and internal workings. It is designed to help you understand the basics of power system modeling and analysis using `pandapower`.

#### pipeline_dnn.ipynb
This notebook demonstrates the implementation of a deep neural network (DNN) pipeline. It includes data preprocessing, model training, and evaluation steps. It is intended to guide you through the process of building and deploying a DNN model for your project.
