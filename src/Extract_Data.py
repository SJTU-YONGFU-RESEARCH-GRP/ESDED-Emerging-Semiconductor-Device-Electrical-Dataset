import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from scipy.interpolate import splprep, splev
import argparse
import os
import pymysql
import logging
import zipfile
from Excel_json import generate_json

def configure_options():
    parser = argparse.ArgumentParser(description= 'Data Extractor')
    parser.add_argument('--id', dest = 'ID', action = 'store', help = 'ID')
    parser.add_argument('--input',dest = 'figure', action = 'store', help = 'name of input figure')
    parser.add_argument('--output', dest = 'OUTPUT', action = 'store', help = 'Output directory')
    parser.add_argument('--HSV', dest = 'HSV', action = 'store', help = 'HSV')
    parser.add_argument('--highX', dest ='highX', action = 'store', help = 'maximum of real x range')
    parser.add_argument('--lowX', dest = 'lowX', action ='store', help = 'minimum of real x range')
    parser.add_argument('--highY', dest = 'highY', action = 'store', help = 'maximum of real y range')
    parser.add_argument('--lowY', dest = 'lowY', action = 'store', help = 'minimum of real y range')
    parser.add_argument('--svalue', dest = 'svalue', action = 'store', help = 'Smoothing parameters for spline fitting.')
    parser.add_argument('--step', dest = 'step', action = 'store', help = 'step size of data point')
    parser.add_argument('--dualCurve', dest = 'dualCurve', action = 'store', help = 'The threshold for determining whether to enter dual-line mode (8 pixels by default)')
    parser.add_argument('--dualLine', dest = 'dualLine', action = 'store', help = 'Used for manual input of two-line mode selection')
    parser.add_argument('--Type', dest='Type', action='store', help= 'The dopant type of transistor')
    parser.add_argument('--Node', dest='Node', action='store', help='The process node of transistor')
    parser.add_argument('--Device', dest='Device', action='store', help='The process structure of transistor')
    # parser.add_argument('--Temp', dest='Temp', action='store', help='The operating temperature of transistor')
    parser.add_argument('--Vds', dest='Vds', action='store', help='Operating Vds of transistor')
    parser.add_argument('--L', dest='L', action='store', help='The channel length of transistor')
    parser.add_argument('--W', dest='W', action='store', help='The channel width of transistor')
    # parser.add_argument('--Nfin', dest='Nfin', action='store', default='None', help='The number of fin in FinFET')
    parser.add_argument('--Info', dest='Info', action='store', help='The information in figure')
    parser.add_argument('--Record', dest='Record', nargs=argparse.REMAINDER, action='store', help='The information record in figure')
    parser.add_argument('--Color', dest='Color', type=int, nargs="+", action='store', help='Select color range')
    args = parser.parse_args()
    print(args)
    return args


def calibrate_coordinates(image_coords, img_x_range, img_y_range, real_x_range, real_y_range):
    """
    Convert image coordinates to real-world coordinates.

    Parameters:
    - image_coords: 2D NumPy array containing image coordinates (x, y).
    - img_x_range: Image x coordinate range [xmin, xmax].
    - img_y_range: Image y coordinate range [ymin, ymax].
    - real_x_range: Real x coordinate range [real_x_min, real_x_max].
    - real_y_range: Real y coordinate range [real_y_min, real_y_max].

    Returns:
    - 2D NumPy array of calibrated real-world coordinates (Vgs, Ids).
    """
    # Linear interpolation for x coordinate
    real_x = np.interp(
        image_coords[:, 0],
        [img_x_range[0], img_x_range[1]],
        [real_x_range[0], real_x_range[1]]
    )

    # Logarithmic scaling and inversion for y coordinate
    real_y = 10**(np.interp(
        image_coords[:, 1],
        [img_y_range[0], img_y_range[1]],
        [np.log10(real_y_range[1]), np.log10(real_y_range[0])]
    ))
    
    return np.vstack((real_x, real_y)).T

def save_data_to_excel(logger, color_data, filename="calibrated_data.xlsx"):
    """
    Save the data in color_data to an Excel file, arranged in groups of two columns with an empty column in between.
    Column names are Vgs(1,2,...) and Ids(1,2,...).

    Parameters:
    - color_data: Dictionary containing data for each color, each as a 2D NumPy array.
    - filename: Name of the Excel file to save (default is "calibrated_data.xlsx").
    """
    try:
        # Initialize an empty DataFrame
        data_dict = {}
        col_index = 1
        
        for color_name, data in color_data.items():
            vgs_column = f"Vgs{col_index}"
            ids_column = f"Ids{col_index}"
            
            # Save Vgs and Ids columns
            data_dict[vgs_column] = data[:, 0]
            data_dict[ids_column] = data[:, 1]
            
            # Add an empty column after each group
            data_dict[f"Empty{col_index}"] = [""] * len(data)
            
            col_index += 1

        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Write data to Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="Calibrated Data", index=False)
        
        logger.info(f"Data saved to {filename}")
    except Exception as e:
        logger.info(f"Failed to save Excel file: {e}")
        
def select_data_points(data, step=5):
    """
    Select data points, with the 'step' parameter defining the interval.

    Parameters:
    - data: 2D NumPy array containing all data points.
    - step: Interval for selecting points (default is 5).

    Returns:
    - 2D NumPy array of selected data points.
    """
    return data[::step]

def spline_fit(data, s):
    """
    Perform spline fitting on the data.

    Parameters:
    - data: 2D NumPy array containing the data points to fit.
    - s: Spline smoothing parameter.

    Returns:
    - 2D NumPy array of fitted x and y coordinates.
    """
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]
    tck, u = splprep([sorted_data[:, 0], sorted_data[:, 1]], s=s)
    u_fine = np.linspace(0, 1, 1000)
    fit_x, fit_y = splev(u_fine, tck)
    return np.column_stack((fit_x, fit_y))

# Define HSV color range list
color_ranges_origin = [
    {"name": "Red", "lower": [0, 120, 200], "upper": [10, 255, 255]},      # Red
    {"name": "Yellow", "lower": [10, 100, 100], "upper": [30, 255, 255]}, # Yellow-Orange
    {"name": "Blue", "lower": [91, 50, 100], "upper": [130, 255, 255]},   # Blue (modified)
    {"name": "Green", "lower": [35, 50, 50], "upper": [90, 255, 255]},    # Green
    {"name": "Purple", "lower": [130, 50, 50], "upper": [165, 255, 255]}  # Purple-Pink
]
def image_process(
    logger,
    image_path,
    output_path,
    color_selection=[0, 2],
    real_x_range=[-2, 1.5],
    real_y_range=[10**(-14), 10**(-4)],
    s_value=800,
    step=10,
    dual_curve_threshold=29,  # Threshold for determining dual-line mode
    dual_line_mode_input=None # Manual input for dual-line mode selection
    
):
    """
    Process the image, extract blue data points, calibrate coordinates, save to Excel, and plot.

    Parameters:
    - image_path: Path to the image file.
    - lower_blue: HSV lower bound array for blue detection.
    - upper_blue: HSV upper bound array for blue detection.
    - real_x_range: Real x coordinate range.
    - real_y_range: Real y coordinate range.
    - s_value: Spline fitting smoothing parameter.
    - step: Step size for selecting data points.
    - dual_curve_threshold: Threshold for determining dual-line mode (default 8 pixels).

    Returns:
    - Dictionary containing calibrated and density-reduced data for two curves.
    """
    color_ranges = [color_ranges_origin[i] for i in color_selection]
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binarization to convert image to black and white
    # Black pixels are close to 0, white pixels are close to 255
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Get image dimensions
    rows, cols = binary.shape
    # Count the number of black pixels in each row and column
    row_black_counts = np.sum(binary == 255, axis=1)
    col_black_counts = np.sum(binary == 255, axis=0)
    # Find the row and column with the most black pixels
    # Assume X axis is in the lower half, Y axis is in the left half
    half_rows = rows // 2
    half_cols = cols // 2

    # Find in upper and lower halves separately
    ymax = np.argmax(row_black_counts[half_rows:]) + half_rows
    ymin = np.argmax(row_black_counts[:half_rows])
    
    # Find in left and right halves separately
    xmin = np.argmax(col_black_counts[:half_cols])
    xmax = np.argmax(col_black_counts[half_cols:]) + half_cols
    
    img_x_range = [xmin, xmax]
    img_y_range = [ymin, ymax]
    
    # Use HSV color space for detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Initialize dictionary to store data for each color
    color_data = {}

    # Iterate over color range list and extract pixels for each color
    for color_info in color_ranges:
        color_name = color_info["name"]
        lower_hsv = np.array(color_info["lower"])
        upper_hsv = np.array(color_info["upper"])

        # Generate mask using color range
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        if color_name == "Red":
            mask1 = cv2.inRange(hsv_image, np.array([170, 120, 200]), np.array([179, 255, 255]))
            mask = cv2.bitwise_or(mask, mask1)
        # Get coordinates of all nonzero pixels in the mask
        y_indices, x_indices = np.where(mask == 255)
        color_coords = np.column_stack((x_indices, y_indices))

        # Check if any pixels of this color were detected
        if color_coords.size == 0:
            print(f"No pixels detected for color {color_name}.")
            continue
        # Extract all x and y of the pixels
        all_x = color_coords[:, 0]
        all_y = color_coords[:, 1]

        # Get unique x coordinates and sort them
        unique_x = np.unique(all_x)
        unique_x_sorted = np.sort(unique_x)

        # Calculate y_diff for all x and count how many x meet the condition
        valid_dual_line_count = 0  # Number of x that meet the condition
        for x in unique_x_sorted:
            y_coords = all_y[all_x == x]  # Get all y for current x
            y_diff = np.max(y_coords) - np.min(y_coords)  # Calculate y_diff
            if y_diff > dual_curve_threshold:
                valid_dual_line_count += 1  # Increment count if condition is met

        # Determine whether to select dual-line mode
        if dual_line_mode_input is None:
            # Default: automatically determine dual-line mode
            if valid_dual_line_count > len(unique_x_sorted) / 3.2:  # If more than 1/3.2 of x meet the condition
                dual_line_mode = True
                print("Dual-line mode selected.")
            else:
                dual_line_mode = False
                print("Single-line mode selected.")
        elif dual_line_mode_input == 1:
            dual_line_mode = False
            print("Single-line mode selected.")
        elif dual_line_mode_input == 2:
            dual_line_mode = True
            print("Dual-line mode selected.")
        else:
            raise ValueError("Invalid input, please enter 1 (single-line mode) or 2 (dual-line mode).")
        
        # Initialize lists to store curve data
        data_lower = []
        data_upper = []
        data_single = []  # Data for single-line mode

        # Iterate over each unique x coordinate and process data points
        for x in unique_x_sorted:
            y_coords = all_y[all_x == x]
            y_min = np.min(y_coords)
            y_max = np.max(y_coords)

            if dual_line_mode:
                y_avg = (y_min + y_max) / 2
                if y_max - y_min > 0.6 * dual_curve_threshold:
                    # Split into upper and lower parts
                    y_upper_coords = y_coords[y_coords > y_avg]
                    new_y_max_upper = np.max(y_upper_coords)
                    new_y_min_upper = np.min(y_upper_coords)
                    y_lower_coords = y_coords[y_coords < y_avg]
                    new_y_max_lower = np.max(y_lower_coords)
                    new_y_min_lower = np.min(y_lower_coords) 
                    if new_y_max_upper - new_y_min_upper < 3 * dual_curve_threshold and new_y_max_lower - new_y_min_lower < 3 * dual_curve_threshold:
                        y_avg_upper = (new_y_max_upper + new_y_min_upper) / 2
                        data_upper.append((x, y_avg_upper))
                        y_avg_lower = (new_y_max_lower + new_y_min_lower) / 2
                        data_lower.append((x, y_avg_lower))
                    else:
                        pass
                else:
                    pass
            else:
                # Single-line mode, calculate average y value
                if y_max - y_min < 3 * dual_curve_threshold:
                    y_avg = (y_min + y_max) / 2
                    data_single.append((x, y_avg))
                else:
                    pass

        # Convert to numpy arrays
        data_lower = np.array(data_lower)[5:-5]
        data_upper = np.array(data_upper)[5:-5]
        data_single = np.array(data_single)[5:-5]

        # Visualize the distribution of converted data points
        plt.figure(figsize=(10, 6))

        # If dual-line mode, plot upper and lower curve data points
        if dual_line_mode:
            if len(data_lower) > 0:
                plt.scatter(
                    data_lower[:, 0],
                    data_lower[:, 1],
                    color='red',
                    label='Lower Raw Data'
                )
            if len(data_upper) > 0:
                plt.scatter(
                    data_upper[:, 0],
                    data_upper[:, 1],
                    color='blue',
                    label='Upper Raw Data'
                )
        else:
            if len(data_single) > 0:
                plt.scatter(
                    data_single[:, 0],
                    data_single[:, 1],
                    color='green',
                    label='Single Raw Data'
                )
        plt.gca().invert_yaxis()
        plt.legend()
        plt.xlabel("Image X Coordinate")
        plt.ylabel("Image Y Coordinate")
        plt.title("Extracted Data Points Before Calibration")
        plt.grid(True)
        # plt.show()
        # Check if there are enough data points for spline fitting
        if dual_line_mode:
            if len(data_lower) < 20:
                logger.info("Too few data points for lower curve, cannot perform spline fitting.")
                continue
            if len(data_upper) < 20:
                logger.info("Too few data points for upper curve, cannot perform spline fitting.")
                continue
        else:
            if len(data_single) < 20:
                logger.info("Too few data points for single-line mode, cannot perform spline fitting.")
                continue

        # Perform spline fitting
        try:
            if dual_line_mode:
                # Fit lower curve
                fit_data_lower = spline_fit(data_lower, s_value)
                # Fit upper curve
                fit_data_upper = spline_fit(data_upper, s_value)
            else:
                # Fit single curve
                fit_data_single = spline_fit(data_single, s_value)
        except Exception as e:
            logger.info(f"Spline fitting failed: {e}")
            return None

        # Calibrate coordinates
        if dual_line_mode:
            calibrated_data_lower = calibrate_coordinates(
                fit_data_lower, img_x_range, img_y_range, real_x_range, real_y_range
            )
            calibrated_data_upper = calibrate_coordinates(
                fit_data_upper, img_x_range, img_y_range, real_x_range, real_y_range
            )
        else:
            calibrated_data_single = calibrate_coordinates(
                fit_data_single, img_x_range, img_y_range, real_x_range, real_y_range
            )

        # Reduce data point density
        if dual_line_mode:
            reduced_data_lower = select_data_points(calibrated_data_lower, step=step)
            reduced_data_upper = select_data_points(calibrated_data_upper, step=step)

            # Save upper and lower curve data to color_data in dual-line mode
            if reduced_data_lower.size > 0:
                color_data[f"{color_name}_Lower"] = reduced_data_lower
            if reduced_data_upper.size > 0:
                color_data[f"{color_name}_Upper"] = reduced_data_upper
        else:
            reduced_data_single = select_data_points(calibrated_data_single, step=step)
            
            # Save single-line mode data to color_data
            if reduced_data_single.size > 0:
                color_data[color_name] = reduced_data_single
    # Save data
    save_data_to_excel(logger, color_data, filename=output_path + "/calibrated_data.xlsx")            
    # Plot all data points on one figure
    plt.figure(figsize=(12, 8))

    # Iterate over color_data and plot each color's data
    for color_name, data in color_data.items():
        # Dynamically get color by name
        base_color_name = color_name.split("_")[0]  # Get base color name (remove _Lower or _Upper)
        plot_color = next((info["name"].lower() for info in color_ranges if info["name"] == base_color_name), "black")

        # Plot data points
        plt.scatter(
            data[:, 0],  # X coordinate
            data[:, 1],  # Y coordinate
            label=color_name,  # Legend shows color name
            color=plot_color,
            s=10  # Point size
        )

    # Set legend, title, and axes
    plt.legend()
    plt.tick_params(direction="in", width=3, labelsize=16, pad=10)
    plt.xlabel("Vgs (Real World)", fontsize=18, fontweight='bold')
    plt.ylabel("Ids (Log Scale)", fontsize=18, fontweight='bold')
    plt.title("Real-World Data Points for All Colors", fontsize=18, fontweight='bold')
    plt.gca().spines["top"].set_linewidth(3)
    plt.gca().spines["right"].set_linewidth(3)
    plt.gca().spines["bottom"].set_linewidth(3)
    plt.gca().spines["left"].set_linewidth(3)
    plt.gca().set_facecolor("white")
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(True)

    # Set Y axis to log scale
    plt.yscale("log")
    plt.xlim(real_x_range)  # X axis range
    plt.ylim(real_y_range)  # Y axis range

    plt.savefig(output_path + "/result.png")

    # Return calibrated data
    if dual_line_mode:
        return {"Lower Curve": reduced_data_lower, "Upper Curve": reduced_data_upper}
    else:
        return {"Single Curve": reduced_data_single}


# Example call to image_process function
# calibrated_data = image_process(
#     "figure7.png",
#     lower_blue = np.array([100, 200, 100]),
#     upper_blue = np.array([130, 255, 255]),
#     real_x_range=[-2, 1.5],
#     real_y_range=[10**(-14), 10**(-4)],
#     s_value=800,
#     step=10,
#     dual_curve_threshold=19,  # Set threshold to 19 pixels
#     dual_line_mode_input=None  # For manual dual-line mode selection
# )

# if calibrated_data is not None:
#     print("Image processing and coordinate calibration completed.")


def format_list(input_list):
    # Use list comprehension to process each element
    formatted_elements = [f'"{item}"' if isinstance(item, str) and ' ' in item else str(item) for item in input_list]
    # Join all elements with a space
    return ' '.join(formatted_elements)


if __name__ == "__main__":
    args = configure_options()
    # print (args.Record)
    if not os.path.exists(args.OUTPUT):
        os.makedirs(args.OUTPUT)
        
    logging.basicConfig(level=logging.INFO, filename=args.OUTPUT + '/curve.log', filemode='w')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(console_handler)
    real_x_range = [eval(args.lowX), eval(args.highX)]
    real_y_range = [eval(args.lowY), eval(args.highY)]
    # HSV = args.HSV.split(",")
    # HSV = [int(num) for num in HSV]
    # lowerHSV = np.array([HSV[0]-10, HSV[1]-10, HSV[2]-10] )
    # upperHSV = np.array([HSV[0]+10, HSV[1]+10, HSV[2]+10])
    # print(real_x_range)
    # print(real_y_range)
    color_selection = args.Color
    try:
        calibrated_data = image_process(
            logger,
            args.figure,
            args.OUTPUT,
            color_selection=color_selection,
            real_x_range=real_x_range,
            real_y_range=real_y_range,
            s_value=int(args.svalue),
            step=int(args.step),
            dual_curve_threshold=int(args.dualCurve),
            dual_line_mode_input=None,
        )
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        calibrated_data = None
    try:
        connection = pymysql.connect(
                host="localhost",
                user="root",
                password="sql_yongfu",
                db="device_model",
            )
        print("database connected")
        cursor = connection.cursor()
        SQL = "SELECT Paper FROM transistor WHERE ID = '%s'" % (args.ID)
        cursor.execute(SQL)
        Name = cursor.fetchone()[0]
        
        # Link = cursor.fetchone()[1]
        SQL = "SELECT PaperLink FROM transistor WHERE ID = '%s'" % (args.ID)
        cursor.execute(SQL)
        Link = cursor.fetchone()[0]
        SQL = "SELECT Data FROM transistor WHERE ID = '%s'" % (args.ID)
        cursor.execute(SQL)
        DataType = cursor.fetchone()[0]
        cursor.close()
        connection.commit()
        connection.close()
        # print (Name)
    except:
        print("database connect error")

    if calibrated_data is not None:

        logger.info("Finished figure processing and coordinate calibration")
        try:
            output_file = generate_json(
                logger=logger,
                file='calibrated_data.xlsx',
                Type=args.Type,
                Node=args.Node,
                Device=args.Device,
                Vds=args.Vds,
                L=args.L,
                W=args.W,
                Name=Name,
                Link=Link,
                Info=args.Info,
                Info_record=args.Record,
                output=args.OUTPUT
            )
        except Exception as e:
            logger.error(f"JSON generation failed: {e}")
            output_file = "None"
            output_figure = "None"
        output_figure = args.OUTPUT + '/result.png'
    else:
        output_file = "None"
        output_figure = "None"
    try:
        connection = pymysql.connect(host='localhost',
                                        user='root',
                                        password='sql_yongfu',
                                        db='device_model')
        print('database connected')
        cursor = connection.cursor()
    
        if os.path.isfile(output_file) and os.path.isfile(output_figure):

            SQL = "UPDATE transistor SET RESULTNAME = '{}', RESULTFIGURE = '{}', status = '{}' WHERE ID = '{}'".format(output_file, output_figure, 'Success', args.ID)
        else:
            SQL = "UPDATE transistor SET status = 'Failed' WHERE ID = '{}'".format(args.ID)

        logger.info(SQL)
        cursor.execute(SQL)
        logger.info("database updated")
        cursor.close()
        connection.commit()
        connection.close()
    except:
        print('database connect error')
            
    # Create a DataFrame to store data
    data = {
        'Paper Name': [Name],
        'Paper Link': [Link],
        'Structure': [args.Device],
        'Dopant Type': [args.Type],
        'Data Type': [DataType],
        'Node (nm)': [args.Node],
        'Channel Length (nm)': [args.L],
        'Channel Width (nm)': [args.W],
        'Vds (V)': [args.Vds],
        'Real X Range(mminimum)': [args.lowX],
        'Real X Range(Maximum)': [args.highX],
        'Real Y Range(minimum)': [args.lowY],
        'Real Y Range(Maximum)': [args.highY],
        'Curve Information': args.Info,
        'Information Records': format_list(args.Record),
        'Color Range': format_list(args.Color),

        'S Value': [args.svalue],
        'Step': [args.step],
        'Dual Curve Threshold': [args.dualCurve],
        'Dual Line': [args.dualLine],
        'Image File': [os.path.basename(args.figure).split('.')[0]],
    }
    # print (args.Color)
    df = pd.DataFrame(data)

    # Save DataFrame to Excel file
    excel_file_path = os.path.join(args.OUTPUT, 'input.xlsx')
    df.to_excel(excel_file_path, index=False)

    logger.info(f"Excel file generated at: {excel_file_path}")

    zip_figure_path = os.path.join(args.OUTPUT, 'figure.zip')
    with zipfile.ZipFile(zip_figure_path, 'w') as zipf:
        zipf.write(args.figure, arcname=os.path.basename(args.figure))

    zip_file_path = os.path.join(args.OUTPUT, 'input.zip')
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(os.path.join(args.OUTPUT, 'input.xlsx'), arcname='input.xlsx')
        zipf.write(zip_figure_path, arcname='figure.zip')
