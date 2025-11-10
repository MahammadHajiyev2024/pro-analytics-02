"""Module 3: Data Prep with DataScrubber


File: src/analytics_project/data_prep.py.
"""

"""
src/analytics_project/data_prep.py

This script uses the DataScrubber class to clean all CSV files:
- sales_data.csv
- customers_data.csv
- products_data.csv

It reads from data/raw/ and writes cleaned files to data/prepared/

Usage:
    python src/analytics_project/data_prep.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import DataScrubber and logger
from src.analytics_project.data_scrubber import DataScrubber
from src.analytics_project.utils_logger import logger

# Define paths
DATA_DIR = project_root / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PREPARED_DATA_DIR = DATA_DIR / "prepared"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PREPARED_DATA_DIR.mkdir(exist_ok=True)


def read_csv_file(file_name: str) -> pd.DataFrame:
    """
    Read a CSV file from the raw data directory.

    Args:
        file_name (str): Name of the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    file_path = RAW_DATA_DIR / file_name
    logger.info(f"Reading file: {file_path}")

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"Columns: {', '.join(df.columns.tolist())}")

    return df


def save_csv_file(df: pd.DataFrame, file_name: str) -> None:
    """
    Save a DataFrame to the prepared data directory.

    Args:
        df (pd.DataFrame): DataFrame to save.
        file_name (str): Name of the output CSV file.
    """
    output_path = PREPARED_DATA_DIR / file_name
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to: {output_path}")
    logger.info(f"Final shape: {df.shape}")


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sales data using DataScrubber.

    Args:
        df (pd.DataFrame): Raw sales DataFrame.

    Returns:
        pd.DataFrame: Cleaned sales DataFrame.
    """
    logger.info("=" * 70)
    logger.info("CLEANING SALES DATA")
    logger.info("=" * 70)

    initial_shape = df.shape
    logger.info(f"Initial shape: {initial_shape}")

    # Check data consistency before cleaning
    scrubber = DataScrubber(df)
    consistency_before = scrubber.check_data_consistency_before_cleaning()
    logger.info(
        f"Null values before cleaning:\n{consistency_before['null_counts'][consistency_before['null_counts'] > 0]}"
    )
    logger.info(f"Duplicate rows before cleaning: {consistency_before['duplicate_count']}")

    # Clean the data
    scrubber = DataScrubber(df)

    # Step 1: Remove duplicates
    scrubber.remove_duplicate_records()
    logger.info(f"After removing duplicates: {scrubber.get_dataframe().shape}")

    # Step 2: Handle special values in numeric columns
    df_temp = scrubber.get_dataframe()

    # Replace '?' with NaN in SaleAmount
    if 'SaleAmount' in df_temp.columns:
        df_temp['SaleAmount'] = pd.to_numeric(df_temp['SaleAmount'], errors='coerce')

    # Replace 'free' with 0 in Shipping
    if 'Shipping' in df_temp.columns:
        df_temp['Shipping'] = df_temp['Shipping'].replace('free', '0')
        df_temp['Shipping'] = pd.to_numeric(df_temp['Shipping'], errors='coerce')

    # Update scrubber with cleaned data
    scrubber = DataScrubber(df_temp)

    # Step 3: Handle missing values
    # Fill CampaignID with 0 (no campaign)
    df_temp = scrubber.get_dataframe()
    if 'CampaignID' in df_temp.columns:
        df_temp['CampaignID'] = df_temp['CampaignID'].fillna(0)

    # Fill Shipping with median
    if 'Shipping' in df_temp.columns:
        median_shipping = df_temp['Shipping'].median()
        df_temp['Shipping'] = df_temp['Shipping'].fillna(median_shipping)
        logger.info(f"Filled missing Shipping with median: {median_shipping}")

    # Drop rows with missing critical values
    critical_columns = [
        'TransactionID',
        'SaleDate',
        'CustomerID',
        'ProductID',
        'StoreID',
        'SaleAmount',
    ]
    existing_critical = [col for col in critical_columns if col in df_temp.columns]
    df_temp = df_temp.dropna(subset=existing_critical)
    logger.info(f"After removing rows with missing critical values: {df_temp.shape}")

    # Update scrubber
    scrubber = DataScrubber(df_temp)

    # Step 4: Format State column to uppercase
    if 'State' in df_temp.columns:
        scrubber.format_column_strings_to_upper_and_trim('State')
        logger.info("Formatted State column to uppercase")

    # Step 5: Remove outliers using IQR method
    df_temp = scrubber.get_dataframe()

    # Remove negative values
    if 'SaleAmount' in df_temp.columns:
        df_temp = df_temp[df_temp['SaleAmount'] >= 0]
    if 'Shipping' in df_temp.columns:
        df_temp = df_temp[df_temp['Shipping'] >= 0]

    # IQR for SaleAmount
    if 'SaleAmount' in df_temp.columns:
        Q1 = df_temp['SaleAmount'].quantile(0.25)
        Q3 = df_temp['SaleAmount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        scrubber = DataScrubber(df_temp)
        scrubber.filter_column_outliers('SaleAmount', lower_bound, upper_bound)
        df_temp = scrubber.get_dataframe()
        logger.info(f"Removed SaleAmount outliers. Range: [{lower_bound:.2f}, {upper_bound:.2f}]")

    # Step 6: Convert date column
    if 'SaleDate' in df_temp.columns:
        df_temp['SaleDate'] = pd.to_datetime(df_temp['SaleDate'], errors='coerce')
        logger.info("Converted SaleDate to datetime")

    # Step 7: Sort by TransactionID
    if 'TransactionID' in df_temp.columns:
        df_temp = df_temp.sort_values('TransactionID').reset_index(drop=True)

    final_shape = df_temp.shape
    rows_removed = initial_shape[0] - final_shape[0]
    logger.info(f"Final shape: {final_shape}")
    logger.info(f"Rows removed: {rows_removed} ({rows_removed / initial_shape[0] * 100:.2f}%)")

    return df_temp


def clean_customers_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean customers data using DataScrubber.

    Args:
        df (pd.DataFrame): Raw customers DataFrame.

    Returns:
        pd.DataFrame: Cleaned customers DataFrame.
    """
    logger.info("=" * 70)
    logger.info("CLEANING CUSTOMERS DATA")
    logger.info("=" * 70)

    initial_shape = df.shape
    logger.info(f"Initial shape: {initial_shape}")

    # Initialize scrubber
    scrubber = DataScrubber(df)

    # Check consistency
    consistency_before = scrubber.check_data_consistency_before_cleaning()
    logger.info(
        f"Null values before cleaning:\n{consistency_before['null_counts'][consistency_before['null_counts'] > 0]}"
    )
    logger.info(f"Duplicate rows before cleaning: {consistency_before['duplicate_count']}")

    # Step 1: Remove duplicates based on CustomerID
    df_temp = scrubber.get_dataframe()
    df_temp = df_temp.drop_duplicates(subset=['CustomerID'], keep='first')
    logger.info(f"After removing duplicates: {df_temp.shape}")

    # Step 2: Format string columns
    scrubber = DataScrubber(df_temp)

    if 'Name' in df_temp.columns:
        scrubber.format_column_strings_to_upper_and_trim('Name')

    if 'Region' in df_temp.columns:
        scrubber.format_column_strings_to_upper_and_trim('Region')

    logger.info("Formatted string columns to uppercase")

    # Step 3: Handle missing values
    df_temp = scrubber.get_dataframe()

    # Drop rows with missing CustomerID or Name
    critical_columns = ['CustomerID', 'Name']
    existing_critical = [col for col in critical_columns if col in df_temp.columns]
    df_temp = df_temp.dropna(subset=existing_critical)

    # Fill missing Region with 'UNKNOWN'
    if 'Region' in df_temp.columns:
        df_temp['Region'] = df_temp['Region'].fillna('UNKNOWN')

    # Fill missing numeric columns with median or 0
    if 'NumberOfPurshases' in df_temp.columns:
        df_temp['NumberOfPurshases'] = df_temp['NumberOfPurshases'].fillna(0)

    if 'ShoppingFrequency' in df_temp.columns:
        df_temp['ShoppingFrequency'] = df_temp['ShoppingFrequency'].fillna(0)

    logger.info(f"After handling missing values: {df_temp.shape}")

    # Step 4: Parse JoinDate
    if 'JoinDate' in df_temp.columns:
        df_temp['JoinDate'] = pd.to_datetime(df_temp['JoinDate'], errors='coerce')
        logger.info("Converted JoinDate to datetime")

    # Step 5: Remove outliers from numeric columns
    if 'NumberOfPurshases' in df_temp.columns:
        Q1 = df_temp['NumberOfPurshases'].quantile(0.25)
        Q3 = df_temp['NumberOfPurshases'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - 1.5 * IQR)  # Can't be negative
        upper_bound = Q3 + 1.5 * IQR

        df_temp = df_temp[
            (df_temp['NumberOfPurshases'] >= lower_bound)
            & (df_temp['NumberOfPurshases'] <= upper_bound)
        ]
        logger.info(
            f"Removed NumberOfPurshases outliers. Range: [{lower_bound:.2f}, {upper_bound:.2f}]"
        )

    # Step 6: Sort by CustomerID
    if 'CustomerID' in df_temp.columns:
        df_temp = df_temp.sort_values('CustomerID').reset_index(drop=True)

    final_shape = df_temp.shape
    rows_removed = initial_shape[0] - final_shape[0]
    logger.info(f"Final shape: {final_shape}")
    logger.info(f"Rows removed: {rows_removed} ({rows_removed / initial_shape[0] * 100:.2f}%)")

    return df_temp


def clean_products_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean products data using DataScrubber.

    Args:
        df (pd.DataFrame): Raw products DataFrame.

    Returns:
        pd.DataFrame: Cleaned products DataFrame.
    """
    logger.info("=" * 70)
    logger.info("CLEANING PRODUCTS DATA")
    logger.info("=" * 70)

    initial_shape = df.shape
    logger.info(f"Initial shape: {initial_shape}")

    # Initialize scrubber
    scrubber = DataScrubber(df)

    # Check consistency
    consistency_before = scrubber.check_data_consistency_before_cleaning()
    logger.info(
        f"Null values before cleaning:\n{consistency_before['null_counts'][consistency_before['null_counts'] > 0]}"
    )
    logger.info(f"Duplicate rows before cleaning: {consistency_before['duplicate_count']}")

    # Step 1: Remove duplicates based on ProductID
    df_temp = scrubber.get_dataframe()
    df_temp = df_temp.drop_duplicates(subset=['ProductID'], keep='first')
    logger.info(f"After removing duplicates: {df_temp.shape}")

    # Step 2: Format string columns
    scrubber = DataScrubber(df_temp)

    if 'ProductName' in df_temp.columns:
        scrubber.format_column_strings_to_upper_and_trim('ProductName')

    if 'Category' in df_temp.columns:
        scrubber.format_column_strings_to_upper_and_trim('Category')

    logger.info("Formatted string columns to uppercase")

    # Step 3: Handle missing values
    df_temp = scrubber.get_dataframe()

    # Drop rows with missing ProductID or ProductName
    critical_columns = ['ProductID', 'ProductName']
    existing_critical = [col for col in critical_columns if col in df_temp.columns]
    df_temp = df_temp.dropna(subset=existing_critical)

    # Fill missing Category with 'UNCATEGORIZED'
    if 'Category' in df_temp.columns:
        df_temp['Category'] = df_temp['Category'].fillna('UNCATEGORIZED')

    # Fill missing Price with median
    if 'Price' in df_temp.columns:
        df_temp['Price'] = pd.to_numeric(df_temp['Price'], errors='coerce')
        median_price = df_temp['Price'].median()
        df_temp['Price'] = df_temp['Price'].fillna(median_price)
        logger.info(f"Filled missing Price with median: {median_price}")

    logger.info(f"After handling missing values: {df_temp.shape}")

    # Step 4: Remove outliers from Price
    if 'Price' in df_temp.columns:
        # Remove negative prices
        df_temp = df_temp[df_temp['Price'] >= 0]

        Q1 = df_temp['Price'].quantile(0.25)
        Q3 = df_temp['Price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - 1.5 * IQR)
        upper_bound = Q3 + 1.5 * IQR

        df_temp = df_temp[(df_temp['Price'] >= lower_bound) & (df_temp['Price'] <= upper_bound)]
        logger.info(f"Removed Price outliers. Range: [{lower_bound:.2f}, {upper_bound:.2f}]")

    # Step 5: Sort by ProductID
    if 'ProductID' in df_temp.columns:
        df_temp = df_temp.sort_values('ProductID').reset_index(drop=True)

    final_shape = df_temp.shape
    rows_removed = initial_shape[0] - final_shape[0]
    logger.info(f"Final shape: {final_shape}")
    logger.info(f"Rows removed: {rows_removed} ({rows_removed / initial_shape[0] * 100:.2f}%)")

    return df_temp


def main():
    """
    Main function to clean all CSV files.
    """
    logger.info("=" * 70)
    logger.info("STARTING DATA PREPARATION FOR ALL FILES")
    logger.info("=" * 70)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Raw data directory: {RAW_DATA_DIR}")
    logger.info(f"Prepared data directory: {PREPARED_DATA_DIR}")

    # Define files to process
    files_to_process = [
        {
            'input': 'sales_data.csv',
            'output': 'sales_cleanedwithScrubber.csv',
            'cleaner': clean_sales_data,
        },
        {
            'input': 'customers_data.csv',
            'output': 'customers_cleanedwithScrubber.csv',
            'cleaner': clean_customers_data,
        },
        {
            'input': 'products_data.csv',
            'output': 'products_cleanedwithScrubber.csv',
            'cleaner': clean_products_data,
        },
    ]

    # Process each file
    for file_info in files_to_process:
        try:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Processing: {file_info['input']}")
            logger.info(f"{'=' * 70}")

            # Read raw data
            df = read_csv_file(file_info['input'])

            # Clean data
            cleaned_df = file_info['cleaner'](df)

            # Save cleaned data
            save_csv_file(cleaned_df, file_info['output'])

            logger.info(f"✓ Successfully processed {file_info['input']}")

        except FileNotFoundError as e:
            logger.warning(f"⚠ Skipping {file_info['input']}: {e}")
        except Exception as e:
            logger.error(f"✗ Error processing {file_info['input']}: {e}")
            raise

    logger.info("\n" + "=" * 70)
    logger.info("DATA PREPARATION COMPLETED FOR ALL FILES")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
