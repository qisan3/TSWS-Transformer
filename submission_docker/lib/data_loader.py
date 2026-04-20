"""
Data Loader Class for PHM Data Challenge

This module defines the DataLoader class, which provides utility 
methods to load and process both controller and sensor data for 
machine tool wear analysis.

Class
-----
DataLoader
    Handles path resolution, loading controller data, and loading 
    sensor data aligned with controller cut times.

Example
-------
    loader = DataLoader(
        controller_data_path="Controller_Data",
        sensor_data_path="Sensor_Data"
    )

    controller_df = loader.get_controller_data(1, 5)
    sensor_df = loader.get_sensor_data(1, 5)
"""

import os
import pandas as pd
import zipfile


class DataLoader:
    """
    Data Loader for PHM Data Challenge datasets.

    Attributes
    ----------
    controller_data_path : str
        Path to the Controller_Data directory.
    sensor_data_path : str
        Path to the Sensor_Data directory.
    set_available : list[int]
        List of valid set numbers available for analysis.
    """

    def __init__(self,
                 controller_data_path: str,
                 sensor_data_path: str,
                 set_available: list[int] = None):
        """
        Initialize the DataLoader.

        Args
        ----
        controller_data_path : str
            Path to the Controller_Data directory.
        sensor_data_path : str
            Path to the Sensor_Data directory.
        set_available : list[int], optional
            List of valid dataset set numbers. Defaults to
            [1, 2, 3].
        """
        self.controller_data_path = controller_data_path
        self.sensor_data_path = sensor_data_path
        self.set_available = set_available or [1, 2, 3]

    def _get_path(self, set_no: int, cut_no: int, channel: str):
        """
        Construct the file path for controller or sensor data.

        Args
        ----
        set_no : int
            Dataset set number (e.g., 1, 2, 3, ...).
        cut_no : int
            Cut number (1–26, except evalset 02 starts at 2).
        channel : str
            'c' for controller, 's' for sensor.

        Returns
        -------
        str or bool
            Full file path if valid, otherwise False.
        """
        check_flag = [0, 0]

        # Validate set number
        if set_no in self.set_available:
            check_flag[0] = 1

        # Validate cut number range
        if set_no != 2 and 1 <= cut_no <= 26:
            check_flag[1] = 1
        if set_no == 2 and 2 <= cut_no <= 26:
            check_flag[1] = 1

        if channel == 'c':  # Controller data
            if sum(check_flag) == 2:
                return os.path.join(
                    self.controller_data_path,
                    f'evalset_{set_no:02d}',
                    f'Cut_{cut_no:02d}.csv'
                )
            return False

        if channel == 's':  # Sensor data (zipped by cut ranges)
            if sum(check_flag) == 2:
                if cut_no == 1:
                    return os.path.join(self.sensor_data_path, f'evalset_{set_no:02d}', 'Part_01_1_1.zip')
                elif 2 <= cut_no <= 6:
                    return os.path.join(self.sensor_data_path, f'evalset_{set_no:02d}', 'Part_02_2_6.zip')
                elif 7 <= cut_no <= 11:
                    return os.path.join(self.sensor_data_path, f'evalset_{set_no:02d}', 'Part_03_7_11.zip')
                elif 12 <= cut_no <= 16:
                    return os.path.join(self.sensor_data_path, f'evalset_{set_no:02d}', 'Part_04_12_16.zip')
                elif 17 <= cut_no <= 21:
                    return os.path.join(self.sensor_data_path, f'evalset_{set_no:02d}', 'Part_05_17_21.zip')
                elif 22 <= cut_no <= 26:
                    return os.path.join(self.sensor_data_path, f'evalset_{set_no:02d}', 'Part_06_22_26.zip')
            return False

    def get_controller_data(self, set_no: int, cut_no: int) -> pd.DataFrame:
        """
        Load controller data for a given set and cut.

        Args
        ----
        set_no : int
            Dataset set number.
        cut_no : int
            Cut number.

        Returns
        -------
        pd.DataFrame
            Controller data with timestamp columns parsed.
            Returns an empty DataFrame if not available.
        """
        path = self._get_path(set_no, cut_no, 'c')
        if path:
            df = pd.read_csv(path)

            # Convert relevant columns to datetime
            for col in ['timestamp', 'start_cut', 'end_cut', 'start_step', 'end_step']:
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")

            return df
        return pd.DataFrame([])

    def get_sensor_data(self, set_no: int, cut_no: int) -> pd.DataFrame:
        """
        Load and filter sensor data for a given set and cut.

        Sensor data is stored inside zip archives grouped by cut ranges.
        Data is filtered using start and end times from the corresponding
        controller data.

        Args
        ----
        set_no : int
            Dataset set number.
        cut_no : int
            Cut number.

        Returns
        -------
        pd.DataFrame
            Sensor data aligned with controller cut duration.
            Returns an empty DataFrame if not available.
        """
        s_path = self._get_path(set_no, cut_no, 's')
        print(s_path)
        if s_path:
            # Load controller data to get cut start/end times
            c_df = self.get_controller_data(set_no, cut_no)
            if c_df.empty:
                return pd.DataFrame([])

            s_start = c_df['start_cut'].dropna().iloc[0]
            s_end = c_df['end_cut'].dropna().iloc[0]

            # Read sensor CSV from zip archive
            with zipfile.ZipFile(s_path) as z:
                csv_name = z.namelist()[0]  # One CSV per zip
                with z.open(csv_name) as f:
                    s_df = pd.read_csv(f)

            # Parse datetime and filter to match controller cut duration
            s_df['Date/Time'] = pd.to_datetime(
                s_df['Date/Time'],
                format="%Y-%m-%d %H:%M:%S.%f",
                errors="coerce"
            )
            s_df = s_df[(s_df['Date/Time'] >= s_start) & (s_df['Date/Time'] <= s_end)].copy()

            return s_df
        return pd.DataFrame([])


