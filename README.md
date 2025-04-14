# ESDED - Emerging Semiconductor Device Electrical Dataset

This repository contains the Emerging Semiconductor Device Electrical Dataset (ESDED), a comprehensive collection of electrical characteristics data for various emerging semiconductor devices. The dataset includes detailed I-V characteristics and device parameters for Gate-All-Around (GAA) FETs, Nanosheet FETs, and Complementary FETs (CFETs) from published research papers.

## Dataset Structure

The dataset is organized into three main device categories:
- `GAA/`: Gate-All-Around FET data
- `Nanosheet/`: Nanosheet FET data
- `CFET/`: Complementary FET data

Each device folder contains multiple subdirectories, where each subdirectory represents a specific device from a published paper. The naming convention follows the pattern `[PaperID]_[FigureID]`.


| Structure | of Devices |
|-----------|--------------------------|
| GAA       | 70                       |
| Nanosheet | 51                       |
| CFET      |  12                      |

## Data Structure

Each file includes metadata such as PDK name, device name, simulator, and model. The `Record` section is a list, in which each entry includes the `Vgs` and the corresponding `Ids` under different dimensions (i.e., `W`, `L`), simulation conditions (i.e., `Temp` and `Corner`), and voltage bias (`Vds`).

| Name       | Description                                | Type             |
|------------|--------------------------------------------|------------------|
| Type       | Type of transistor                         | String           |
| Node       | Technology node in nanometers              | Float            |
| Device     | Name of MOSFET device                      | String           |
| Vds        | Drain-source voltage in volts              | Float            |
| Title      | Title of the paper                         | String           |
| DOI        | DOI of the paper                           | String           |
| W          | Width of the MOSFET in nanometers          | Integer          |
| L          | Length of the MOSFET in nanometers         | Integer          |
| Vgs        | Gate-source voltage in volts               | List of float    |
| Ids        | Drain current measured in amperes          | List of float    |


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