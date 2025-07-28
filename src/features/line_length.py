import numpy as np
from typing import List, Union, Tuple

def compute_line_length(signal: np.ndarray, window_size: int = 256) -> float:
    """
    Вычисляет Line Length (LL) для одного канала сигнала.
    
    Args:
        signal (np.ndarray): Сигнал одного канала
        window_size (int): Размер окна (по умолчанию 256 для 1 секунды при 256 Гц)
    
    Returns:
        float: Значение Line Length
    """
    return np.sum(np.abs(np.diff(signal)))

def extract_line_length_features(
    signals: np.ndarray,
    sfreq: float = 256.0,
    window_size: float = 1.0
) -> np.ndarray:
    """
    Извлекает признаки Line Length из многоканального сигнала ЭЭГ.
    
    Args:
        signals (np.ndarray): Многоканальный сигнал формы (n_channels, n_samples)
        sfreq (float): Частота дискретизации
        window_size (float): Размер окна в секундах
        
    Returns:
        np.ndarray: Матрица признаков Line Length формы (n_channels,)
    """
    n_channels, n_samples = signals.shape
    samples_per_window = int(sfreq * window_size)
    
    # Проверка размера входных данных
    if n_samples < samples_per_window:
        raise ValueError(f"Длина сигнала ({n_samples}) меньше размера окна ({samples_per_window})")
    
    features = np.zeros(n_channels)
    for i in range(n_channels):
        features[i] = compute_line_length(signals[i], samples_per_window)
    
    return features

def extract_line_length_windows(
    signals: np.ndarray,
    sfreq: float = 256.0,
    window_size: float = 10.0,
    step_size: float = 1.0
) -> Tuple[np.ndarray, List[float]]:
    """
    Извлекает признаки Line Length с использованием скользящего окна.
    
    Args:
        signals (np.ndarray): Многоканальный сигнал формы (n_channels, n_samples)
        sfreq (float): Частота дискретизации
        window_size (float): Размер окна в секундах
        step_size (float): Шаг окна в секундах
        
    Returns:
        Tuple[np.ndarray, List[float]]: 
            - Матрица признаков формы (n_windows, n_channels)
            - Список временных меток для каждого окна
    """
    n_channels, n_samples = signals.shape
    samples_per_window = int(sfreq * window_size)
    samples_per_step = int(sfreq * step_size)
    
    # Вычисляем количество окон
    n_windows = (n_samples - samples_per_window) // samples_per_step + 1
    
    features = np.zeros((n_windows, n_channels))
    timestamps = []
    
    for i in range(n_windows):
        start_idx = i * samples_per_step
        end_idx = start_idx + samples_per_window
        
        # Извлекаем признаки для текущего окна
        window_signals = signals[:, start_idx:end_idx]
        features[i] = extract_line_length_features(window_signals, sfreq, window_size)
        
        # Сохраняем временную метку (центр окна)
        timestamps.append((start_idx + end_idx) / (2 * sfreq))
    
    return features, timestamps

if __name__ == "__main__":
    # Пример использования
    # Генерируем синтетические данные для тестирования
    n_channels, n_samples = 18, 2560  # 10 секунд при 256 Гц
    test_signals = np.random.randn(n_channels, n_samples)
    
    # Извлекаем признаки
    features, times = extract_line_length_windows(test_signals)
    
    print(f"Размер матрицы признаков: {features.shape}")
    print(f"Количество временных меток: {len(times)}")
    print(f"Первые значения признаков:\n{features[0]}")
    print(f"Временные метки: {times[:5]}") 