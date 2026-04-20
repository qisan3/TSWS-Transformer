import os
import pandas as pd
import zipfile
import numpy as np
import matplotlib.pyplot as plt

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
        self.set_available = set_available or [1, 2, 3, 4, 5, 6]

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
        check_flag = [0, 1]

        # Validate set number
        if set_no in self.set_available:
            check_flag[0] = 1


        if channel == 'c':  # Controller data
            if sum(check_flag) == 2:
                return os.path.join(
                    self.controller_data_path,
                    f'trainset_{set_no:02d}',
                    f'Cut_{cut_no:02d}.csv'
                )
            return False

        if channel == 's':  # Sensor data (zipped by cut ranges)
            if sum(check_flag) == 2:
                if cut_no == 1:
                    return os.path.join(self.sensor_data_path, f'trainset_{set_no:02d}', 'Part_01_1_1.zip')
                elif 2 <= cut_no <= 6:
                    return os.path.join(self.sensor_data_path, f'trainset_{set_no:02d}', 'Part_02_2_6.zip')
                elif 7 <= cut_no <= 11:
                    return os.path.join(self.sensor_data_path, f'trainset_{set_no:02d}', 'Part_03_7_11.zip')
                elif 12 <= cut_no <= 16:
                    return os.path.join(self.sensor_data_path, f'trainset_{set_no:02d}', 'Part_04_12_16.zip')
                elif 17 <= cut_no <= 21:
                    return os.path.join(self.sensor_data_path, f'trainset_{set_no:02d}', 'Part_05_17_21.zip')
                elif 22 <= cut_no <= 26:
                    return os.path.join(self.sensor_data_path, f'trainset_{set_no:02d}', 'Part_06_22_26.zip')
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


if __name__ == '__main__':

    # --- 推荐的循环加载方式 (不使用 tqdm) ---

    # 1. 初始化 DataLoader
    # 确保路径 "Controller_Data" 和 "Sensor_Data"
    # 与你的Python脚本在同一目录，否则请提供绝对路径
    loader = DataLoader("Controller_Data", "Sensor_Data")

    # 2. 定义你要循环的范围
    set_range = range(1, 7)  # Set 1, 2, 3, 4, 5, 6
    cut_range = range(1, 27)  # Cut 1, 2, ..., 26

    # 3. 创建空字典来存储所有数据
    all_controller_data = {}
    all_sensor_data = {}

    total_files = len(set_range) * len(cut_range)
    print(f"开始加载 {total_files} 个数据文件...")

    current_file_count = 0

    # 4. 使用嵌套循环遍历所有 set 和 cut
    for set_no in set_range:
        for cut_no in cut_range:

            current_file_count += 1

            # 5. 创建一个唯一的键 (key)，就像你想要的变量名
            data_key = f"trainset{set_no}_cut{cut_no}"

            # 打印当前进度
            print(f"正在处理: {data_key} ({current_file_count} / {total_files})...")

            # 6. 调用你的函数来加载数据
            controller_df = loader.get_controller_data(set_no, cut_no)
            sensor_df = loader.get_sensor_data(set_no, cut_no)

            # 7. 检查数据是否为空，如果非空则存入字典
            if not controller_df.empty:
                all_controller_data[data_key] = controller_df
            else:
                print(f"Warning: 未找到 {data_key} 的控制器数据")

            if not sensor_df.empty:
                all_sensor_data[data_key] = sensor_df
            else:
                print(f"Warning: 未找到 {data_key} 的传感器数据")

    print("\n--- 所有数据加载完成! ---")

    # --- 如何使用这些数据 ---

    print(f"总共加载了 {len(all_controller_data)} 个控制器数据文件。")
    print(f"总共加载了 {len(all_sensor_data)} 个传感器数据文件。")
#    # 示例：访问 trainset 1, cut 5 的数据
# key_to_access = "trainset1_cut5"
# if key_to_access in all_controller_data:
#         print(f"\n--- 访问 {key_to_access} 的控制器数据 (前5行) ---")
#         print(all_controller_data[key_to_access].head())
#
#     # 示例：访问 trainset 6, cut 26 的数据
# key_to_access = "trainset6_cut26"
# if key_to_access in all_sensor_data:
#         print(f"\n--- 访问 {key_to_access} 的传感器数据 (前5行) ---")
#         print(all_sensor_data[key_to_access].head())