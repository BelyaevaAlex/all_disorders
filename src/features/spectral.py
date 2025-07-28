import numpy as np
from scipy import signal
from typing import List, Dict, Tuple, Optional

# Определение частотных диапазонов ЭЭГ
FREQ_BANDS = {
    'delta': (0.5, 4),    # Дельта-ритм
    'theta': (4, 8),      # Тета-ритм
    'alpha': (8, 13),     # Альфа-ритм
    'beta_low': (13, 20), # Низкочастотный бета-ритм
    'beta_high': (20, 30),# Высокочастотный бета-ритм
    'gamma_low': (30, 45),# Низкочастотный гамма-ритм
    'gamma_high': (45, 80)# Высокочастотный гамма-ритм
}

def compute_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: Tuple[float, float]
) -> float:
    """
    Вычисляет мощность сигнала в заданном частотном диапазоне.
    
    Args:
        freqs (np.ndarray): Массив частот
        psd (np.ndarray): Массив значений спектральной плотности мощности
        freq_range (Tuple[float, float]): Диапазон частот (min_freq, max_freq)
    
    Returns:
        float: Мощность сигнала в заданном диапазоне
    """
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    return np.mean(psd[mask]) if np.any(mask) else 0.0

def extract_spectral_features(
    signal_window: np.ndarray,
    sfreq: float = 256.0,
    window_sec: float = 1.0,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Извлекает спектральные признаки из одного окна сигнала.
    
    Args:
        signal_window (np.ndarray): Окно сигнала
        sfreq (float): Частота дискретизации
        window_sec (float): Длина окна в секундах
        freq_bands (Dict[str, Tuple[float, float]], optional): 
            Словарь частотных диапазонов
    
    Returns:
        np.ndarray: Вектор спектральных признаков
    """
    if freq_bands is None:
        freq_bands = FREQ_BANDS
        
    # Вычисляем PSD с помощью метода Уэлча
    freqs, psd = signal.welch(
        signal_window,
        fs=sfreq,
        nperseg=int(sfreq * window_sec),
        noverlap=int(sfreq * window_sec * 0.5)
    )
    
    # Извлекаем признаки для каждого частотного диапазона
    features = []
    for band_name, freq_range in freq_bands.items():
        band_power = compute_band_power(freqs, psd, freq_range)
        features.append(band_power)
    
    return np.array(features)

def extract_spectral_features_multichannel(
    signals: np.ndarray,
    sfreq: float = 256.0,
    window_sec: float = 1.0,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Извлекает спектральные признаки из многоканального сигнала.
    
    Args:
        signals (np.ndarray): Многоканальный сигнал формы (n_channels, n_samples)
        sfreq (float): Частота дискретизации
        window_sec (float): Длина окна в секундах
        freq_bands (Dict[str, Tuple[float, float]], optional): 
            Словарь частотных диапазонов
    
    Returns:
        np.ndarray: Матрица признаков формы (n_channels * n_bands,)
    """
    if freq_bands is None:
        freq_bands = FREQ_BANDS
        
    n_channels = signals.shape[0]
    n_bands = len(freq_bands)
    features = np.zeros(n_channels * n_bands)
    
    for i in range(n_channels):
        channel_features = extract_spectral_features(
            signals[i],
            sfreq=sfreq,
            window_sec=window_sec,
            freq_bands=freq_bands
        )
        features[i * n_bands:(i + 1) * n_bands] = channel_features
    
    return features

def extract_spectral_windows(
    signals: np.ndarray,
    sfreq: float = 256.0,
    window_size: float = 10.0,
    step_size: float = 1.0,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[np.ndarray, List[float]]:
    """
    Извлекает спектральные признаки с использованием скользящего окна.
    
    Args:
        signals (np.ndarray): Многоканальный сигнал формы (n_channels, n_samples)
        sfreq (float): Частота дискретизации
        window_size (float): Размер окна в секундах
        step_size (float): Шаг окна в секундах
        freq_bands (Dict[str, Tuple[float, float]], optional): 
            Словарь частотных диапазонов
    
    Returns:
        Tuple[np.ndarray, List[float]]:
            - Матрица признаков формы (n_windows, n_channels * n_bands)
            - Список временных меток для каждого окна
    """
    if freq_bands is None:
        freq_bands = FREQ_BANDS
        
    n_channels, n_samples = signals.shape
    samples_per_window = int(sfreq * window_size)
    samples_per_step = int(sfreq * step_size)
    
    # Вычисляем количество окон
    n_windows = (n_samples - samples_per_window) // samples_per_step + 1
    
    # Инициализируем массивы для хранения результатов
    n_features = n_channels * len(freq_bands)
    features = np.zeros((n_windows, n_features))
    timestamps = []
    
    for i in range(n_windows):
        start_idx = i * samples_per_step
        end_idx = start_idx + samples_per_window
        
        # Извлекаем признаки для текущего окна
        window_signals = signals[:, start_idx:end_idx]
        features[i] = extract_spectral_features_multichannel(
            window_signals,
            sfreq=sfreq,
            window_sec=1.0,  # Используем 1-секундные окна для спектрального анализа
            freq_bands=freq_bands
        )
        
        # Сохраняем временную метку (центр окна)
        timestamps.append((start_idx + end_idx) / (2 * sfreq))
    
    return features, timestamps

if __name__ == "__main__":
    # Пример использования
    # Генерируем синтетические данные для тестирования
    n_channels, n_samples = 18, 2560  # 10 секунд при 256 Гц
    test_signals = np.random.randn(n_channels, n_samples)
    
    # Извлекаем спектральные признаки
    features, times = extract_spectral_windows(test_signals)
    
    print(f"Размер матрицы признаков: {features.shape}")
    print(f"Количество временных меток: {len(times)}")
    print(f"Количество признаков на канал: {len(FREQ_BANDS)}")
    print(f"Первые значения признаков:\n{features[0][:len(FREQ_BANDS)]}")  # Показываем признаки первого канала
    print(f"Временные метки: {times[:5]}") 