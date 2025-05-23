# ESDED - Emerging Semiconductor Device Electrical Dataset

The Emerging Semiconductor Device Electrical Dataset (ESDED) is a comprehensive, curated collection of electrical characteristics for a wide range of emerging semiconductor devices. This dataset is designed to support research, modeling, and machine learning applications in the field of advanced semiconductor technology.

## Overview

ESDED provides detailed I-V characteristics and device parameters for:
- **Gate-All-Around (GAA) FETs**
- **Nanosheet FETs**
- **Complementary FETs (CFETs)**

All data is extracted from published research papers, ensuring high quality and relevance.

## Data Collection and Curation Process

All data in ESDED is manually extracted from peer-reviewed publications. Each device entry is cross-checked for accuracy, and metadata is standardized for consistency. Where possible, raw data is digitized from published figures using state-of-the-art extraction tools. The curation process includes:
- Careful selection of high-quality, relevant publications.
- Manual digitization and extraction of I-V curves and device parameters.
- Standardization of metadata fields and units.
- Validation and cross-checking to ensure accuracy and consistency across the dataset.

## Dataset Organization

The dataset is organized into three main directories:
- `GAA/` — Gate-All-Around FET data (70 devices)
- `Nanosheet/` — Nanosheet FET data (51 devices)
- `CFET/` — Complementary FET data (12 devices)

Each subdirectory within these folders corresponds to a specific device, named as `[PaperID]_[FigureID]`, referencing the source publication and figure.

## Directory and File Structure Example

```
ESDED/
├── GAA/
│   ├── Paper123_Fig2/
│   │   └── device_data.json
│   └── ...
├── Nanosheet/
│   └── ...
├── CFET/
│   └── ...
```
Each `device_data.json` contains all records and metadata for a specific device.

## Data Format

Each device file is in JSON format and contains:
- **Metadata**: Device type, technology node, device name, and source paper information.
- **Measurement Data**: A list of records, each with:
  - `Vgs`: Gate-source voltage (V)
  - `Ids`: Corresponding drain current (A)
  - Device dimensions (`W`, `L`), simulation conditions (`Temp`, `Corner`), and voltage bias (`Vds`).

### Detailed Data Schema

- `Type` (string): Type of transistor (e.g., "GAA", "Nanosheet", "CFET")
- `Node` (float): Technology node in nanometers
- `Device` (string): Device name
- `Vds` (float): Drain-source voltage in volts
- `Title` (string): Title of the source paper
- `DOI` (string): DOI of the source paper
- `W` (integer): Device width in nanometers
- `L` (integer): Device length in nanometers
- `Record` (list): List of measurement sweeps, each with:
  - `Vgs` (list of float): Gate-source voltage values (V)
  - `Ids` (list of float): Drain current values (A)
  - `Temp` (integer, optional): Temperature in Kelvin (e.g., 300 for room temperature)
  - `Corner` (string, optional): Process corner, e.g., "TT" (typical-typical), "SS" (slow-slow), "FF" (fast-fast)

### Field Descriptions

| Field   | Description                                 | Type             |
|---------|---------------------------------------------|------------------|
| Type    | Type of transistor                          | String           |
| Node    | Technology node (nm)                        | Float            |
| Device  | Device name                                 | String           |
| Vds     | Drain-source voltage (V)                    | Float            |
| Title   | Title of the source paper                   | String           |
| DOI     | DOI of the source paper                     | String           |
| W       | Device width (nm)                           | Integer          |
| L       | Device length (nm)                          | Integer          |
| Vgs     | Gate-source voltage values (V)              | List of float    |
| Ids     | Drain current values (A)                    | List of float    |
| Temp    | Temperature (K, optional)                   | Integer          |
| Corner  | Process corner (optional)                   | String           |

## Example

```json
{
  "Type": "GAA",
  "Node": 5,
  "Device": "SampleDevice",
  "Vds": 0.7,
  "Title": "Example Paper Title",
  "DOI": "10.1234/example.doi",
  "W": 30,
  "L": 20,
  "Record": [
    {
      "Vgs": [0.0, 0.1, 0.2, ...],
      "Ids": [0.0, 1e-7, 2e-7, ...],
      "Temp": 300,
      "Corner": "TT"
    }
  ]
}
```

## How to Use the Dataset

You can easily load and analyze the dataset using Python. Below is a simple example for loading a device JSON file and accessing its contents:

```python
import json

with open('GAA/Paper123_Fig2/device_data.json') as f:
    data = json.load(f)

print('Device:', data['Device'])
print('Vgs:', data['Record'][0]['Vgs'])
print('Ids:', data['Record'][0]['Ids'])
```

You can use libraries such as `matplotlib` to visualize the I-V characteristics:

```python
import matplotlib.pyplot as plt

vgs = data['Record'][0]['Vgs']
ids = data['Record'][0]['Ids']
plt.plot(vgs, ids)
plt.xlabel('Vgs (V)')
plt.ylabel('Ids (A)')
plt.title(f"I-V Curve for {data['Device']}")
plt.show()
```

## Applications

This dataset is ideal for:
- Device modeling and simulation
- Machine learning model training
- Device performance benchmarking
- Research and educational purposes

## License

This dataset is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use ESDED in your research, please cite the original papers from which the data was collected. 
```
Feiyang Shen, Ruoyu Tang, Qing Zhang, Chao Wang, Yuhang Zhang, Kain Lu Low, Yongfu Li, "ESDED - Emerging Semiconductor Device Electrical Dataset", IEEE Data Description, May, 2025, doi:10.21227/e4zb-x710
```
Full citation information is included in each device's JSON file.
