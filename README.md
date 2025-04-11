# ESDED - Emerging Semiconductor Device Electrical Dataset

This repository contains the Emerging Semiconductor Device Electrical Dataset (ESDED), a comprehensive collection of electrical characteristics data for various emerging semiconductor devices. The dataset includes detailed I-V characteristics and device parameters for Gate-All-Around (GAA) FETs, Nanosheet FETs, and Complementary FETs (CFETs) from published research papers.

## Dataset Structure

The dataset is organized into three main device categories:
- `GAA/`: Gate-All-Around FET data
- `Nanosheet/`: Nanosheet FET data
- `CFET/`: Complementary FET data

Each device folder contains multiple subdirectories, where each subdirectory represents a specific device from a published paper. The naming convention follows the pattern `[PaperID]_[DeviceID]`.

## Data File Structure

Each device directory contains:
1. A JSON file (`[DeviceID].json`) containing the electrical characteristics data
2. Visualization files (PNG format) showing the device characteristics

### JSON Data Format

The JSON files contain the following information:
```json
{
    "Type": "Device type (e.g., Negative)",
    "Node": "Technology node",
    "Device": "Device type (e.g., GAA)",
    "Vds": "Drain-source voltage",
    "L": "Channel length (nm)",
    "W": "Channel width (nm)",
    "Paper Name": "Title of the source paper",
    "Paper Link": "Link to the paper",
    "Records": [
        {
            "Vds": "Drain-source voltage (V)",
            "Vgs": ["List of gate-source voltages (V)"],
            "Ids": ["List of drain currents (A)"]
        }
    ]
}
```

## Data Collection

The dataset is compiled from published research papers, with each entry including:
- Device electrical characteristics (I-V curves)
- Device physical parameters (dimensions)
- Source paper information
- Visualization of the characteristics

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