"""
scripts/data_preparation/prepare_sales.py

This script reads data from the data/raw folder, cleans the data,
and writes the cleaned version to the data/prepared folder.

Tasks:
- Remove duplicates
- Handle missing values
- Remove outliers
- Ensure consistent formatting

"""

#####################################
# Import Modules at the Top
#####################################

# Import from Python Standard Library
import pathlib
import sys

# Import from external packages (requires a virtual environment)
import pandas as pd

# Ensure project root is in sys.path for local imports (now 3 parents are needed)
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

# Import local modules (e.g. utils/logger.py)
from utils.logger import logger

# Optional: Use a data_scrubber module for common data cleaning tasks
from utils.data_scrubber import DataScrubber


# Constants
SCRIPTS_DATA_PREP_DIR: pathlib.Path = (
    pathlib.Path(__file__).resolve().parent
)  # Directory of the current script
SCRIPTS_DIR: pathlib.Path = SCRIPTS_DATA_PREP_DIR.parent
PROJECT_ROOT: pathlib.Path = SCRIPTS_DIR.parent
DATA_DIR: pathlib.Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: pathlib.Path = DATA_DIR / "raw"
PREPARED_DATA_DIR: pathlib.Path = DATA_DIR / "prepared"  # place to store prepared data


# Ensure the directories exist or create them
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PREPARED_DATA_DIR.mkdir(exist_ok=True)

#####################################
# Define Functions - Reusable blocks of code / instructions
#####################################

# TODO: Complete this by implementing functions based on the logic in the other scripts


def read_raw_data(file_name: str) -> pd.DataFrame:
    """
    Read raw data from CSV.

    Args:
        file_name (str): Name of the CSV file to read.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    logger.info(f"FUNCTION START: read_raw_data with file_name={file_name}")
    file_path = RAW_DATA_DIR.joinpath(file_name)
    logger.info(f"Reading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")

    # TODO: OPTIONAL Add data profiling here to understand the dataset
    # Suggestion: Log the datatypes of each column and the number of unique values
    # Example:
    # logger.info(f"Column datatypes: \n{df.dtypes}")
    # logger.info(f"Number of unique values: \n{df.nunique()}")

    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize column names.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    logger.info(f"FUNCTION START: clean_column_names")
    original_columns = df.columns.tolist()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.capitalize()

    # Log if any column names changed
    changed_columns = [
        f"{old} -> {new}" for old, new in zip(original_columns, df.columns) if old != new
    ]
    if changed_columns:
        logger.info(f"Cleaned column names: {', '.join(changed_columns)}")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    logger.info(f"FUNCTION START: remove_duplicates with dataframe shape={df.shape}")
    initial_count = len(df)

    # Remove duplicates based on TransactionID (should be unique)
    df = df.drop_duplicates(subset=['Transactionid'], keep='first')

    # Also check for duplicates across all columns
    df = df.drop_duplicates(keep='first')

    removed_count = initial_count - len(df)
    logger.info(f"Removed {removed_count} duplicate rows")
    logger.info(f"{len(df)} records remaining after removing duplicates.")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    logger.info(f"FUNCTION START: handle_missing_values with dataframe shape={df.shape}")
    initial_count = len(df)

    # Log missing values before handling
    missing_summary = df.isnull().sum()
    logger.info(f"Missing values before handling:\n{missing_summary[missing_summary > 0]}")

    # Handle '?' as missing value in SaleAmount
    if 'Saleamount' in df.columns:
        df['Saleamount'] = pd.to_numeric(df['Saleamount'], errors='coerce')

    # Handle 'free' as 0 in Shipping column
    if 'Shipping' in df.columns:
        df['Shipping'] = df['Shipping'].replace('free', '0')
        df['Shipping'] = pd.to_numeric(df['Shipping'], errors='coerce')

    # Handle missing CampaignID - fill with 0 (no campaign)
    if 'Campaignid' in df.columns:
        df['Campaignid'] = df['Campaignid'].fillna(0)

    # For critical columns, remove rows with missing values
    critical_columns = ['Transactionid', 'Saledate', 'Customerid', 'Productid', 'Storeid']
    existing_critical = [col for col in critical_columns if col in df.columns]
    df = df.dropna(subset=existing_critical)

    # For SaleAmount, we can either drop or impute
    # Business decision: drop rows with missing SaleAmount as it's a key metric
    if 'Saleamount' in df.columns:
        df = df.dropna(subset=['Saleamount'])

    # Fill remaining missing Shipping values with median
    if 'Shipping' in df.columns:
        median_shipping = df['Shipping'].median()
        df['Shipping'] = df['Shipping'].fillna(median_shipping)
        logger.info(f"Filled missing Shipping values with median: {median_shipping}")

    removed_count = initial_count - len(df)
    logger.info(f"Removed {removed_count} rows due to missing critical values")
    logger.info(f"{len(df)} records remaining after handling missing values.")

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers based on business rules and statistical methods.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    logger.info(f"FUNCTION START: remove_outliers with dataframe shape={df.shape}")
    initial_count = len(df)

    # Remove negative values (business rule: these shouldn't exist)
    if 'Saleamount' in df.columns:
        df = df[df['Saleamount'] >= 0]

    if 'Shipping' in df.columns:
        df = df[df['Shipping'] >= 0]

    # IQR method for SaleAmount
    if 'Saleamount' in df.columns:
        Q1 = df['Saleamount'].quantile(0.25)
        Q3 = df['Saleamount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        before_outlier_removal = len(df)
        df = df[(df['Saleamount'] >= lower_bound) & (df['Saleamount'] <= upper_bound)]
        logger.info(f"Removed {before_outlier_removal - len(df)} SaleAmount outliers (IQR method)")
        logger.info(f"SaleAmount range: [{lower_bound:.2f}, {upper_bound:.2f}]")

    # IQR method for Shipping
    if 'Shipping' in df.columns:
        Q1 = df['Shipping'].quantile(0.25)
        Q3 = df['Shipping'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        before_outlier_removal = len(df)
        df = df[(df['Shipping'] >= lower_bound) & (df['Shipping'] <= upper_bound)]
        logger.info(f"Removed {before_outlier_removal - len(df)} Shipping outliers (IQR method)")
        logger.info(f"Shipping range: [{lower_bound:.2f}, {upper_bound:.2f}]")

    removed_count = initial_count - len(df)
    logger.info(f"Total removed {removed_count} outlier rows")
    logger.info(f"{len(df)} records remaining after removing outliers.")

    return df


def ensure_consistent_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure consistent data formatting across the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with consistent formatting.
    """
    logger.info(f"FUNCTION START: ensure_consistent_formatting")

    # Convert SaleDate to datetime
    if 'Saledate' in df.columns:
        df['Saledate'] = pd.to_datetime(df['Saledate'], errors='coerce')
        logger.info(f"Converted Saledate to datetime format")

    # Ensure numeric columns are proper numeric types
    numeric_columns = [
        'Transactionid',
        'Customerid',
        'Productid',
        'Storeid',
        'Campaignid',
        'Saleamount',
        'Shipping',
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Standardize State codes to uppercase
    if 'State' in df.columns:
        df['State'] = df['State'].str.upper().str.strip()

    # Sort by TransactionID for consistency
    if 'Transactionid' in df.columns:
        df = df.sort_values('Transactionid').reset_index(drop=True)

    logger.info(f"Ensured consistent formatting across all columns")

    return df


def save_prepared_data(df: pd.DataFrame, file_name: str) -> None:
    """
    Save the prepared DataFrame to CSV.

    Args:
        df (pd.DataFrame): Prepared DataFrame.
        file_name (str): Name of the output CSV file.
    """
    logger.info(f"FUNCTION START: save_prepared_data with file_name={file_name}")
    output_path = PREPARED_DATA_DIR.joinpath(file_name)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved prepared data to {output_path}")
    logger.info(f"Final dataframe shape: {df.shape}")


#####################################
# Define Main Function - The main entry point of the script
#####################################


def main() -> None:
    """
    Main function for processing data.
    """
    logger.info("==================================")
    logger.info("STARTING prepare_sales_data.py")
    logger.info("==================================")

    logger.info(f"Root         : {PROJECT_ROOT}")
    logger.info(f"data/raw     : {RAW_DATA_DIR}")
    logger.info(f"data/prepared: {PREPARED_DATA_DIR}")
    logger.info(f"scripts      : {SCRIPTS_DIR}")

    input_file = "sales_data.csv"
    output_file = "sales_prepared.csv"

    # Read raw data
    df = read_raw_data(input_file)

    # Record original shape
    original_shape = df.shape

    # Log initial dataframe information
    logger.info(f"Initial dataframe columns: {', '.join(df.columns.tolist())}")
    logger.info(f"Initial dataframe shape: {df.shape}")

    # Clean column names
    df = clean_column_names(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Remove outliers
    df = remove_outliers(df)

    # Ensure consistent formatting
    df = ensure_consistent_formatting(df)

    # Save prepared data
    save_prepared_data(df, output_file)

    logger.info("==================================")
    logger.info(f"Original shape: {original_shape}")
    logger.info(f"Cleaned shape:  {df.shape}")
    logger.info(
        f"Data reduction: {original_shape[0] - df.shape[0]} rows removed ({((original_shape[0] - df.shape[0]) / original_shape[0] * 100):.2f}%)"
    )
    logger.info("==================================")
    logger.info("FINISHED prepare_sales_data.py")
    logger.info("==================================")


#####################################
# Conditional Execution Block
# Ensures the script runs only when executed directly
# This is a common Python convention.
#####################################

if __name__ == "__main__":
    main()
