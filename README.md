# ESDED - Emerging Semiconductor Device Electrical Dataset

This repository contains the Emerging Semiconductor Device Electrical Dataset (ESDED), a comprehensive collection of electrical characteristics data for various emerging semiconductor devices. The dataset includes detailed I-V characteristics and device parameters for Gate-All-Around (GAA) FETs, Nanosheet FETs, and Complementary FETs (CFETs) from published research papers.

## Dataset Structure

The dataset is organized into three main device categories:
- `GAA/`: Gate-All-Around FET data
- `Nanosheet/`: Nanosheet FET data
- `CFET/`: Complementary FET data

Each device folder contains multiple subdirectories, where each subdirectory represents a specific device from a published paper. The naming convention follows the pattern `[PaperID]_[DeviceID]`.

## Data File Naming Convention

The dataset is in `MESD` folder. The data is recorded with JSON files. Each JSON file corresponds to a device in a PDK. The files are named as `PDK Name`-`Device Name`. MESD includes data from 11 PDKs and 44 types of devices from 350 nm to 3 nm. The PDKs and devices are renamed to avoid leakage of foundry information. 

| Technology | PDK Name | # of Devices (NMOS) | # of Devices (PMOS) |
|------------|----------|-------------------------|--------------------------|
| 3 nm       | N3A      | 1                       | 1                        |
| 7 nm       | N7A      | 3                       | 3                        |
| 15 nm      | N15A     | 1                       | 1                        |
| 40 nm      | N40A     | 3                       | 3                        |
| 45 nm      | N45A     | 1                       | 1                        |
| 45 nm      | N45B     | 3                       | 3                        |
| 55 nm      | N55A     | 3                       | 3                        |
| 90 nm      | N90A     | 3                       | 3                        |
| 180 nm     | N180A    | 1                       | 1                        |
| 180 nm     | N180B    | 2                       | 2                        |
| 350 nm     | N350A    | 1                       | 1                        |

## Data Structure

Each file includes metadata such as PDK name, device name, simulator, and model. The `Record` section is a list, in which each entry includes the `Vgs` and the corresponding `Ids` and `Cgg` under different dimensions (i.e., `W`, `L`, and `Nfin`), simulation conditions (i.e., `Temp` and `Corner`), and voltage bias (`Vds`).

| Name       | Description                                | Type             |
|------------|--------------------------------------------|------------------|
| PDK        | Name of Process Design Kit                 | String           |
| Node       | Technology node in nanometers              | Integer          |
| Device     | Name of MOSFET device                      | String           |
| Type       | Type of MOSFET (NMOS/PMOS)                 | String           |
| Simulator  | Simulator adopted to collect data          | String           |
| Model      | Name of compact model to collect data      | String           |
| Corner     | Process corner                             | String           |
| Temp       | The temperature in degrees Celsius         | Integer          |
| W          | Width of the MOSFET in nanometers          | Integer          |
| L          | Length of the MOSFET in nanometers         | Integer          |
| Nfin       | Number of fins (for FinFET only)           | Integer          |
| Vds        | Drain-source voltage in volts              | Float            |
| Vgs        | Gate-source voltage in volts               | List of float    |
| Ids        | Drain current measured in amperes          | List of float    |
| Cgg        | Gate capacitance in farads                 | List of float    |


## Usage

The dataset can be used for:
- Device modeling and simulation
- Machine learning model training
- Device performance comparison
- Research and educational purposes

## License

This dataset is licensed under the terms of the Creative Commons Attribution 4.0 International License.

## Citation

If you use this dataset in your research, please cite the original papers from which the data was collected. The paper information is included in each device's JSON file.