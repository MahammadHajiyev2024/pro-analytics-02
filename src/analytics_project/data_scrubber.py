"""
utils/data_scrubber.py

Reusable utility class for performing common data cleaning and
preparation tasks on a pandas DataFrame.

This class provides methods for:
- Checking data consistency
- Removing duplicates
- Handling missing values
- Filtering outliers
- Renaming and reordering columns
- Formatting strings
- Parsing date fields

Use this class to perform similar cleaning operations across multiple files.
You are not required to use this class, but it shows how we can organize
reusable data cleaning logic - or you can use the logic examples in your own code.


from utils.data_scrubber import DataScrubber

scrubber = DataScrubber(df)
df = scrubber.remove_duplicate_records().handle_missing_data(fill_value="N/A").get_dataframe()
"""

import io
import pandas as pd
from typing import Dict, Tuple, Union, List, Optional


class DataScrubber:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataScrubber with a DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to be scrubbed.
        """
        self.df = df.copy()  # Create a copy to avoid modifying original

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the current DataFrame.

        Returns:
            pd.DataFrame: The current state of the DataFrame.
        """
        return self.df

    def check_data_consistency_before_cleaning(self) -> Dict[str, Union[pd.Series, int]]:
        """
        Check data consistency before cleaning by calculating counts of null and duplicate entries.

        Returns:
            dict: Dictionary with counts of null values and duplicate rows.
        """
        null_counts = self.df.isnull().sum()
        duplicate_count = self.df.duplicated().sum()
        return {'null_counts': null_counts, 'duplicate_count': duplicate_count}

    def check_data_consistency_after_cleaning(self) -> Dict[str, Union[pd.Series, int]]:
        """
        Check data consistency after cleaning to ensure there are no null or duplicate entries.

        Returns:
            dict: Dictionary with counts of null values and duplicate rows, expected to be zero for each.
        """
        null_counts = self.df.isnull().sum()
        duplicate_count = self.df.duplicated().sum()
        assert null_counts.sum() == 0, "Data still contains null values after cleaning."
        assert duplicate_count == 0, "Data still contains duplicate records after cleaning."
        return {'null_counts': null_counts, 'duplicate_count': duplicate_count}

    def convert_column_to_new_data_type(self, column: str, new_type: type) -> 'DataScrubber':
        """
        Convert a specified column to a new data type.

        Parameters:
            column (str): Name of the column to convert.
            new_type (type): The target data type (e.g., 'int', 'float', 'str').

        Returns:
            DataScrubber: Returns self for method chaining.

        Raises:
            ValueError: If the specified column not found in the DataFrame.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column name '{column}' not found in the DataFrame.")

        try:
            self.df[column] = self.df[column].astype(new_type)
        except Exception as e:
            raise ValueError(f"Could not convert column '{column}' to type {new_type}: {str(e)}")

        return self

    def drop_columns(self, columns: List[str]) -> 'DataScrubber':
        """
        Drop specified columns from the DataFrame.

        Parameters:
            columns (list): List of column names to drop.

        Returns:
            DataScrubber: Returns self for method chaining.

        Raises:
            ValueError: If a specified column is not found in the DataFrame.
        """
        for column in columns:
            if column not in self.df.columns:
                raise ValueError(f"Column name '{column}' not found in the DataFrame.")
        self.df = self.df.drop(columns=columns)
        return self

    def filter_column_outliers(
        self, column: str, lower_bound: Union[float, int], upper_bound: Union[float, int]
    ) -> 'DataScrubber':
        """
        Filter outliers in a specified column based on lower and upper bounds.

        Parameters:
            column (str): Name of the column to filter for outliers.
            lower_bound (float or int): Lower threshold for outlier filtering.
            upper_bound (float or int): Upper threshold for outlier filtering.

        Returns:
            DataScrubber: Returns self for method chaining.

        Raises:
            ValueError: If the specified column not found in the DataFrame.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column name '{column}' not found in the DataFrame.")

        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self

    def format_column_strings_to_lower_and_trim(self, column: str) -> 'DataScrubber':
        """
        Format strings in a specified column by converting to lowercase and trimming whitespace.

        Parameters:
            column (str): Name of the column to format.

        Returns:
            DataScrubber: Returns self for method chaining.

        Raises:
            ValueError: If the specified column not found in the DataFrame.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column name '{column}' not found in the DataFrame.")

        self.df[column] = self.df[column].astype(str).str.lower().str.strip()
        return self

    def format_column_strings_to_upper_and_trim(self, column: str) -> 'DataScrubber':
        """
        Format strings in a specified column by converting to uppercase and trimming whitespace.

        Parameters:
            column (str): Name of the column to format.

        Returns:
            DataScrubber: Returns self for method chaining.

        Raises:
            ValueError: If the specified column not found in the DataFrame.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column name '{column}' not found in the DataFrame.")

        self.df[column] = self.df[column].astype(str).str.upper().str.strip()
        return self

    def handle_missing_data(
        self, drop: bool = False, fill_value: Optional[Union[float, int, str]] = None
    ) -> 'DataScrubber':
        """
        Handle missing data in the DataFrame.

        Parameters:
            drop (bool, optional): If True, drop rows with missing values. Default is False.
            fill_value (any, optional): Value to fill in for missing entries if drop is False.

        Returns:
            DataScrubber: Returns self for method chaining.
        """
        if drop:
            self.df = self.df.dropna()
        elif fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self

    def inspect_data(self) -> Tuple[str, str]:
        """
        Inspect the data by providing DataFrame information and summary statistics.

        Returns:
            tuple: (info_str, describe_str), where `info_str` is a string representation of DataFrame.info()
                   and `describe_str` is a string representation of DataFrame.describe().
        """
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        info_str = buffer.getvalue()

        describe_str = self.df.describe().to_string()
        return info_str, describe_str

    def parse_dates_to_add_standard_datetime(self, column: str) -> 'DataScrubber':
        """
        Parse a specified column as datetime format and add it as a new column named 'StandardDateTime'.

        Parameters:
            column (str): Name of the column to parse as datetime.

        Returns:
            DataScrubber: Returns self for method chaining.

        Raises:
            ValueError: If the specified column not found in the DataFrame.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column name '{column}' not found in the DataFrame.")

        self.df['StandardDateTime'] = pd.to_datetime(self.df[column], errors='coerce')
        return self

    def remove_duplicate_records(self) -> 'DataScrubber':
        """
        Remove duplicate rows from the DataFrame.

        Returns:
            DataScrubber: Returns self for method chaining.
        """
        self.df = self.df.drop_duplicates()
        return self

    def rename_columns(self, column_mapping: Dict[str, str]) -> 'DataScrubber':
        """
        Rename columns in the DataFrame based on a provided mapping.

        Parameters:
            column_mapping (dict): Dictionary where keys are old column names and values are new names.

        Returns:
            DataScrubber: Returns self for method chaining.

        Raises:
            ValueError: If a specified column is not found in the DataFrame.
        """
        for old_name in column_mapping.keys():
            if old_name not in self.df.columns:
                raise ValueError(f"Column '{old_name}' not found in the DataFrame.")

        self.df = self.df.rename(columns=column_mapping)
        return self

    def reorder_columns(self, columns: List[str]) -> 'DataScrubber':
        """
        Reorder columns in the DataFrame based on the specified order.

        Parameters:
            columns (list): List of column names in the desired order.

        Returns:
            DataScrubber: Returns self for method chaining.

        Raises:
            ValueError: If a specified column is not found in the DataFrame.
        """
        for column in columns:
            if column not in self.df.columns:
                raise ValueError(f"Column name '{column}' not found in the DataFrame.")
        self.df = self.df[columns]
        return self
