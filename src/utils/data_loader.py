import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import os
import sys
import torch

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features.spectral import extract_spectral_windows
from src.features.line_length import extract_line_length_windows

class EEGDataLoader:
    def __init__(
        self,
        data: str | pd.DataFrame,
        window_size: float = 10.0,
        step_size: float = 1.0,
        sfreq: Optional[float] = None
    ):
        """
        Инициализация загрузчика данных ЭЭГ.
        
        Args:
            data (str | pd.DataFrame): Путь к CSV файлу с данными или DataFrame
            window_size (float): Размер окна в секундах
            step_size (float): Шаг окна в секундах
            sfreq (float, optional): Частота дискретизации
        """
        self.window_size = window_size
        self.step_size = step_size
        
        # Загружаем данные
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data
            
        self.sfreq = sfreq if sfreq is not None else self.df['sfreq'].iloc[0]
        
        # Определяем каналы ЭЭГ (исключаем служебные колонки и проблемные каналы)
        excluded_columns = [
            'duration_sec', 'event', 'file_path', 'label',
            'n_channels', 'patient_id', 'sfreq', 'Status',
            'A1', 'A2', 'Fpz'  # Исключаем проблемные каналы
        ]
        self.eeg_channels = [col for col in self.df.columns if col not in excluded_columns]
        print(f"Используемые каналы ЭЭГ: {', '.join(self.eeg_channels)}")
        
        # Создаем маппинг меток в числовые значения
        self.label_mapping = {label: idx for idx, label in enumerate(self.df['label'].unique())}
        self.label_mapping_inv = {idx: label for label, idx in self.label_mapping.items()}
        self.num_classes = len(self.label_mapping)
        
        # Преобразуем метки в числовые значения
        self.labels = np.array([self.label_mapping[label] for label in self.df['label']])
        
        # Загружаем все данные сразу
        features, labels, record_ids, _ = self.load_data()  # Игнорируем mapping
        self.features = features
        self.all_labels = labels
        self.record_ids = record_ids
        self.num_features = self.features.shape[2]  # Сохраняем количество признаков
        
    def __len__(self) -> int:
        """
        Возвращает количество окон в датасете.
        """
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Возвращает окно данных по индексу.
        
        Args:
            idx (int): Индекс окна
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Тензор признаков
                - Тензор метки
                - Тензор ID записи
        """
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.all_labels[idx]])
        record_id = torch.LongTensor([self.record_ids[idx]])
        return features, label.squeeze(), record_id.squeeze()
        
    def _load_eeg_segment(self, row: pd.Series) -> np.ndarray:
        """
        Загружает сегмент ЭЭГ из строки датафрейма.
        
        Args:
            row (pd.Series): Строка датафрейма
            
        Returns:
            np.ndarray: Матрица сигналов ЭЭГ формы (n_channels, n_samples)
        """
        signals = []
        for channel in self.eeg_channels:
            signal = np.array([float(x) for x in row[channel].strip('[]').split(',')])
            signals.append(signal)
        return np.array(signals)
    
    def extract_features(
        self,
        signals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Извлекает признаки из сигналов ЭЭГ.
        
        Args:
            signals (np.ndarray): Матрица сигналов ЭЭГ
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[float]]:
                - Спектральные признаки
                - Признаки Line Length
                - Временные метки
        """
        # Извлекаем спектральные признаки
        spectral_features, timestamps = extract_spectral_windows(
            signals,
            sfreq=self.sfreq,
            window_size=self.window_size,
            step_size=self.step_size
        )
        
        # Извлекаем признаки Line Length
        ll_features, _ = extract_line_length_windows(
            signals,
            sfreq=self.sfreq,
            window_size=self.window_size,
            step_size=self.step_size
        )
        
        # Объединяем признаки, сохраняя временное измерение
        # spectral_features: (windows, channels * bands)
        # ll_features: (windows, channels)
        # Преобразуем в (windows, timesteps, features)
        num_windows = spectral_features.shape[0]
        num_channels = len(self.eeg_channels)
        timesteps = int(self.window_size * self.sfreq)  # количество временных точек в окне
        
        # Преобразуем признаки в формат (windows, timesteps, features)
        # Для каждой временной точки используем те же признаки окна
        spectral_reshaped = np.repeat(spectral_features[:, np.newaxis, :], timesteps, axis=1)
        ll_reshaped = np.repeat(ll_features[:, np.newaxis, :], timesteps, axis=1)
        
        # Объединяем все признаки
        combined_features = np.concatenate([spectral_reshaped, ll_reshaped], axis=2)
        
        return combined_features, ll_features, timestamps
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Загружает все данные и извлекает признаки.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
                - Матрица признаков
                - Вектор меток
                - Вектор ID записей для каждого окна
                - Словарь маппинга меток
        """
        all_features = []
        all_labels = []
        all_record_ids = []  # Добавляем список для хранения ID записей
        skipped_records = 0
        total_records = len(self.df)
        
        for record_num, (idx, row) in enumerate(self.df.iterrows(), 1):
            print(f"Обработка записи {record_num}/{total_records}")
            
            try:
                # Проверяем все каналы перед обработкой
                valid_record = True
                expected_length = None
                
                for channel in self.eeg_channels:
                    try:
                        signal = np.array([float(x) for x in row[channel].strip('[]').split(',')])
                        if expected_length is None:
                            expected_length = len(signal)
                        elif len(signal) != expected_length:
                            print(f"Пропуск записи {record_num}: разная длина сигналов ({len(signal)} != {expected_length})")
                            valid_record = False
                            break
                    except Exception as e:
                        print(f"Пропуск записи {record_num}: ошибка в канале {channel} - {str(e)}")
                        valid_record = False
                        break
                
                if not valid_record:
                    skipped_records += 1
                    continue
                
                # Загружаем сигналы
                signals = self._load_eeg_segment(row)
                
                # Извлекаем признаки
                features, _, _ = self.extract_features(signals)
                
                # Добавляем метки для каждого окна
                labels = np.full(len(features), self.label_mapping[row['label']])
                # Добавляем ID записи для каждого окна
                record_ids = np.full(len(features), idx)
                
                all_features.append(features)
                all_labels.append(labels)
                all_record_ids.append(record_ids)
                
            except Exception as e:
                print(f"Пропуск записи {record_num} из-за ошибки: {str(e)}")
                skipped_records += 1
                continue
        
        if len(all_features) == 0:
            raise ValueError("Не удалось загрузить ни одной записи!")
        
        # Объединяем все признаки, метки и ID записей
        X = np.vstack(all_features)  # (total_windows, timesteps, features)
        y = np.hstack(all_labels)    # (total_windows,)
        record_ids = np.hstack(all_record_ids)  # (total_windows,)
        
        print(f"\nОбработка завершена:")
        print(f"Всего записей: {total_records}")
        print(f"Пропущено записей: {skipped_records}")
        print(f"Успешно обработано: {total_records - skipped_records}")
        print(f"\nРазмерности данных:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"record_ids shape: {record_ids.shape}")
        
        return X, y, record_ids, self.label_mapping

if __name__ == "__main__":
    # Пример использования
    data_loader = EEGDataLoader(
        csv_path="/home/belyaeva.a/df_open.csv",
        window_size=10.0,
        step_size=1.0
    )
    
    # Загружаем данные
    X, y, label_mapping, record_ids = data_loader.load_data()
    
    print(f"Размер матрицы признаков: {X.shape}")
    print(f"Размер вектора меток: {y.shape}")
    print(f"Маппинг меток: {label_mapping}")
    print(f"Количество классов: {data_loader.num_classes}")
    print(f"Распределение классов:")
    for label, idx in label_mapping.items():
        print(f"{label}: {np.sum(y == idx)}") 