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

## Included Notebooks

### pandapower_init_and_internals.ipynb
This notebook provides an introduction to the `pandapower` library, including its initialization and internal workings. It is designed to help you understand the basics of power system modeling and analysis using `pandapower` in terms of accessing internal states and specify initial states. More detailed tutorial can be found via the [official website](https://www.pandapower.org/)

### pipeline_dnn.ipynb
This notebook demonstrates the implementation of a deep neural network (DNN) pipeline. It includes data preprocessing, model training, and evaluation steps. It is intended to guide you through the process of building and deploying a DNN model in the context of `pandapower` for your project.

## Data
This repository includes four base network and functionality to generate training networks based on parameterization of key parameters.

### Data structure
The data generation is handled by the `net_gen.py` script, which uses the definitions provided in `simple_net/transformer_high_pu.py`,  `simple_net/high_generation_injection.py` `complete_net/high_generation_segmented_grid.py` and `complete_net/transformer_high_pu_segmented_grid.py` to create networks with varying parameters.

#### Base Networks
1. **High Generation Injection Network (`HignGenInjectionNet`)**
   - This network simulates a high generation injection scenario.
   - It consists of:
     - A source bus at 10kV (node_0).
     - A line with 1 ohm resistance connecting to another bus (node_1).
     - A load connected to node_1.
   - Parameters:
     - `vm_pu`: Voltage magnitude at the external grid bus.
     - `p_mw`: Active power generation.
     - `q_mvar`: Reactive power generation.
     - `init_vm_pu`: Initial voltage magnitudes for the power flow calculation.

2. **Transformer High PU Network (`TrafoHighPuNet`)**
   - This network simulates a transformer with high per-unit values.
   - It consists of:
     - Two buses at 10kV.
     - A transformer connecting the two buses.
     - An external grid connected to the first bus.
     - A load connected to the second bus.
   - Parameters:
     - `vm_pu`: Voltage magnitude at the external grid bus.
     - `p_mw`: Active power load.
     - `q_mvar`: Reactive power load.
     - `vn_lv_kv`: Low voltage side nominal voltage of the transformer.
     - `init_vm_pu`: Initial voltage magnitudes for the power flow calculation.

3. **Large Networks with High Generation Injection (`HighGenSegmentedNet`)**
   - This network simulates a high generation injection scenario with multiple segments.
   - It consists of:
     - A source bus at 10kV.
     - Multiple segments with lines and distributed generation.
     - An external grid connected to the first bus.
   - Parameters:
     - `vm_pu`: Voltage magnitude at the external grid bus.
     - `total_p_gen`: Total active power generation distributed across segments.
     - `total_q_gen`: Total reactive power generation distributed across segments.
     - `p_mw`: Active power generation at the last bus.
     - `q_mvar`: Reactive power generation at the last bus.
     - `init_vm_pu`: Initial voltage magnitudes for the power flow calculation.

4. **Large Networks with High PU Transformer (`TrafoHighPuSegmentedNet`)**
   - This network simulates a transformer with high per-unit values and multiple segments.
   - It consists of:
     - Two buses at 10kV.
     - A transformer connecting the two buses.
     - Multiple segments with lines, 10kv bus, and distributed generation.
     - An external grid connected to the first bus.
   - Parameters:
     - `vm_pu`: Voltage magnitude at the external grid bus.
     - `vn_lv_kv`: Low voltage side nominal voltage of the transformer.
     - `total_p_gen`: Total active power generation distributed across segments.
     - `total_q_gen`: Total reactive power generation distributed across segments.
     - `init_vm_pu`: Initial voltage magnitudes for the power flow calculation.

### Data Generation
The `net_gen.py` script provides functions to generate multiple instances of these networks with randomized parameters within specified ranges:

- **`sample_net_high_gen_inj_xs(N)`**:
  - Generates `N` instances of `HignGenInjectionNet` with randomized parameters.
  - Parameter ranges:
    - `VM_PU_RANGE = [0.9, 3.0]`
    - `P_MW_RANGE = [0.0, 1000.0]`
    - `Q_MVAR_RANGE = [0.0, 0.5]`
    - `INIT_VM_PU_MIN = 0.9`
    - `INIT_VM_PU_MAX = 3.0`

- **`sample_net_trafo_high_pu_xs(N)`**:
  - Generates `N` instances of `TrafoHighPuNet` with randomized parameters.
  - Parameter ranges:
    - `VM_PU_RANGE = [0.9, 3.0]`
    - `P_MW_RANGE = [0.0, 1000.0]`
    - `Q_MVAR_RANGE = [0.0, 0.5]`
    - `VN_LV_KV_RANGE = [0.9, 3.0]`
    - `INIT_VM_PU_MIN = 0.9`
    - `INIT_VM_PU_MAX = 3.0`

- **`sample_net_high_gen_segmented_xl(N)`**:
  - Generates `N` instances of `HighGenSegmentedNet` with randomized parameters.
  - Parameter ranges:
    - `VM_PU_RANGE = [0.9, 3.0]`
    - `TOTAL_P_GEN_RANGE = [0.0, 1000.0]`
    - `TOTAL_Q_GEN_RANGE = [0.0, 0.5]`
    - `P_MW_RANGE = [0.0, 1000.0]`
    - `Q_MVAR_RANGE = [0.0, 0.5]`
    - `INIT_VM_PU_MIN = 0.9`
    - `INIT_VM_PU_MAX = 3.0`

- **`sample_net_trafo_high_pu_segmented_xl(N)`**:
  - Generates `N` instances of `TrafoHighPuSegmentedNet` with randomized parameters.
  - Parameter ranges:
    - `VM_PU_RANGE = [0.9, 3.0]`
    - `VN_LV_KV_RANGE = [0.9, 3.0]`
    - `TOTAL_P_GEN_RANGE = [0.0, 1000.0]`
    - `TOTAL_Q_GEN_RANGE = [0.0, 0.5]`
    - `INIT_VM_PU_MIN = 0.9`
    - `INIT_VM_PU_MAX = 3.0`

These functions return lists of network instances that can be used for training or analysis purposes.
