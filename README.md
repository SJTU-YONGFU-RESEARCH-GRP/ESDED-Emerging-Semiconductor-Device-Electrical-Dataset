 # ESDED - Emerging Semiconductor Device Electrical Dataset

The Emerging Semiconductor Device Electrical Dataset (ESDED) is a comprehensive, curated collection of electrical characteristics for a wide range of emerging semiconductor devices. This dataset is designed to support research, modeling, and machine learning applications in the field of advanced semiconductor technology.

## 📊 Dataset Overview

ESDED provides detailed I-V characteristics and device parameters for:
- **Gate-All-Around (GAA) FETs** - Gate-All-Around Field Effect Transistors
- **Nanosheet FETs** - Nanosheet Field Effect Transistors  
- **Complementary FETs (CFETs)** - Complementary Field Effect Transistors
- **Tunnel FETs (TFETs)** - Tunnel Field Effect Transistors

All data is extracted from published research papers, ensuring high quality and relevance.

## 📁 Dataset Organization

The dataset is organized into four main directories:
- `GAA/` — Gate-All-Around FET data (approximately 70 devices)
- `Nanosheet/` — Nanosheet FET data (approximately 51 devices)
- `CFET/` — Complementary FET data (approximately 70 devices)
- `TFET/` — Tunnel FET data (approximately 25 devices)

Each subdirectory corresponds to a specific device, named as `[PaperID]_[FigureID]`, referencing the source publication and figure.

### Directory Structure Example

```
ESDED/
├── GAA/
│   ├── 9903594_74D/
│   │   ├── 9903594_74D.json    # Device data file
│   │   ├── result.png          # Processing result image
│   │   └── 74D.png            # Original image
│   └── ...
├── Nanosheet/
│   └── ...
├── CFET/
│   └── ...
├── TFET/
│   └── ...
└── src/
    ├── Extract_Data.py        # Image data extraction tool
    └── Excel_json.py          # Excel to JSON conversion tool
```

## 📋 Data Format

Each device file is in JSON format and contains:
- **Metadata**: Device type, technology node, device name, and source paper information
- **Measurement Data**: A list of records, each with:
  - `Vgs`: Gate-source voltage (V)
  - `Ids`: Corresponding drain current (A)
  - Device dimensions (`W`, `L`), voltage bias (`Vds`), and other parameters

### Detailed Data Schema

| Field    | Description                           | Type           |
|----------|----------------------------------------|----------------|
| Type     | Type of transistor                     | String         |
| Node     | Technology node (nm)                   | Float          |
| Device   | Device name or structure               | List of String |
| Vds      | Drain-source voltage (V)               | Float          |
| Title    | Title of the source paper              | String         |
| DOI      | DOI or link to the source paper        | String         |
| W        | Device width (nm)                      | Float          |
| L        | Device length (nm)                     | Float          |
| Records  | List of measurement sweeps             | List of Object |
| Vgs      | Gate-source voltage values (V)         | List of Float  |
| Ids      | Drain current values (A)               | List of Float  |
| H/Vds    | Sweep parameters (e.g., height, Vds)   | String/Number  |

### Example JSON Structure

```json
{
    "Type": "Negative",
    "Node": 0.0,
    "Device": ["GAA"],
    "Vds": 0.0,
    "L": 40.0,
    "W": 30.0,
    "Title": "A Gate-All-Around In2O3 Nanoribbon FET With Near 20 mA/µm Drain Current",
    "DOI": "https://ieeexplore.ieee.org/document/9903594",
    "Records": [
        {
            "Vds": "1.7",
            "Vgs": [-2.51, -2.49, -2.48, ...],
            "Ids": [0.072, 0.080, 0.090, ...]
        }
    ]
}
```

## 🛠️ Data Processing Tools

### 1. Image Data Extraction Tool: `src/Extract_Data.py`

This script extracts I-V data from device figure images, calibrates coordinates, and exports to Excel format.

**Main Features:**
- Image preprocessing and coordinate calibration
- Multi-color curve extraction
- Spline fitting and smoothing
- Data point density control
- Dual-curve mode support

**Command Line Parameters:**
```bash
python src/Extract_Data.py --input figure.png --output ./output \
    --highX 1.5 --lowX -2 --highY 1e-4 --lowY 1e-14 \
    --Type GAA --Node 5 --Device SampleDevice \
    --Vds 0.7 --L 20 --W 30 \
    --Info "Vds" --Record "1.7" "0.8" \
    --Color 0 2 --svalue 800 --step 10
```

**Key Parameter Descriptions:**
- `--input`: Input image file
- `--output`: Output directory
- `--highX/--lowX/--highY/--lowY`: Real coordinate axis ranges
- `--svalue`: Spline smoothing parameter
- `--step`: Data point selection step size
- `--Color`: Color range selection
- `--Type/--Node/--Device/--Vds/--L/--W`: Device metadata
- `--Info/--Record`: Sweep information

### 2. Excel to JSON Conversion Tool: `src/Excel_json.py`

Converts Excel files (generated from extracted data) to standardized JSON format.

**Features:**
- Automatic processing of multiple Vgs/Ids data groups
- Metadata standardization
- Record sorting and validation
- Consistent output format

**Usage Notes:**
- Delete the first row in the Excel file before using
- Supports batch processing of multi-curve data

## 📈 Applications

This dataset is ideal for:
- **Device Modeling and Simulation**: Providing validation data for TCAD and SPICE models
- **Machine Learning Training**: Training device characteristic prediction models
- **Performance Benchmarking**: Comparing performance across different device architectures
- **Research and Education**: Semiconductor device characteristic teaching and research

## 📄 License

This dataset is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).



Full citation information is included in each device's JSON file.

## 🤝 Contributing

We welcome contributions through:
- Reporting data errors or issues
- Submitting new device data
- Improving data processing tools
- Enhancing documentation


**The ESDED project is dedicated to providing high-quality, standardized electrical characteristic data for semiconductor device research.**