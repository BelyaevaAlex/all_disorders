import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mutual_info_score
import pandas as pd

class PatternAnalyzer:
    def __init__(self, feature_names: List[str], class_names: List[str]):
        """
        Инициализация анализатора паттернов
        
        Args:
            feature_names: Список названий каналов и частотных диапазонов
            class_names: Список названий классов
        """
        self.feature_names = feature_names
        self.class_names = class_names
        self.patterns_cache = {}
        
    def detect_pattern(
        self,
        data: torch.Tensor,
        window_size: int = 3,
        threshold: float = 0.5
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Обнаруживает паттерны изменений в данных
        
        Args:
            data: Тензор формы (batch_size, sequence_length, features)
            window_size: Размер окна для анализа
            threshold: Порог для определения значимых изменений
            
        Returns:
            Dictionary с паттернами для каждого временного окна
        """
        patterns = {}
        batch_size, seq_len, n_features = data.shape
        
        for t in range(seq_len - window_size + 1):
            window = data[:, t:t+window_size, :]
            
            # Вычисляем изменения в окне
            changes = window[:, -1, :] - window[:, 0, :]
            # Проверяем, является ли changes тензором PyTorch
            if isinstance(changes, torch.Tensor):
                changes = changes.cpu().numpy()
            # Теперь changes точно numpy array, вычисляем среднее
            mean_changes = changes.mean(axis=0)
            
            # Находим значимые изменения
            significant_changes = []
            for idx, (change, feature) in enumerate(zip(mean_changes, self.feature_names)):
                if abs(change) > threshold:
                    significant_changes.append({
                        'feature': feature,
                        'change': float(change),
                        'significance': float(abs(change))
                    })
            
            if significant_changes:
                window_key = f"window_{t}_{t+window_size}"
                patterns[window_key] = significant_changes
                # Сохраняем паттерны в кэш
                self.patterns_cache[window_key] = significant_changes
                
        return patterns
        
    def analyze_class_patterns(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        window_size: int = 3,
        threshold: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """
        Анализирует связь паттернов с классами
        
        Args:
            data: Тензор формы (batch_size, sequence_length, features)
            labels: Метки классов
            window_size: Размер окна для анализа
            threshold: Порог для определения значимых изменений
            
        Returns:
            Dictionary с информацией о связи паттернов с классами
        """
        patterns = self.detect_pattern(data, window_size, threshold)
        class_patterns = {class_name: {} for class_name in self.class_names}
        
        for window_key, changes in patterns.items():
            # Создаем вектор изменений с учетом их величины
            pattern_vector = np.zeros(len(self.feature_names))
            for change in changes:
                idx = self.feature_names.index(change['feature'])
                pattern_vector[idx] = change['change']
                
            # Анализируем связь с каждым классом
            for class_idx, class_name in enumerate(self.class_names):
                class_mask = labels == class_idx
                if class_mask.sum() > 0:
                    # Вычисляем взаимную информацию между паттерном и классом
                    # Дискретизируем значения для mutual_info_score
                    pattern_discrete = np.digitize(pattern_vector, bins=np.linspace(pattern_vector.min(), pattern_vector.max(), 10))
                    mi_score = mutual_info_score(
                        pattern_discrete,
                        class_mask.numpy()
                    )
                    if mi_score > 0:  # Сохраняем только информативные паттерны
                        class_patterns[class_name][window_key] = float(mi_score)
        
        return class_patterns
        
    def get_discriminative_patterns(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        window_size: int = 3,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> Dict[str, List[Dict[str, any]]]:
        """
        Находит наиболее дискриминативные паттерны для каждого класса
        
        Args:
            data: Тензор формы (batch_size, sequence_length, features)
            labels: Метки классов
            window_size: Размер окна для анализа
            top_k: Количество топ паттернов для каждого класса
            threshold: Порог для определения значимых изменений
            
        Returns:
            Dictionary с топ паттернами для каждого класса
        """
        class_patterns = self.analyze_class_patterns(data, labels, window_size, threshold)
        discriminative_patterns = {}
        
        for class_name, patterns in class_patterns.items():
            # Сортируем паттерны по их информативности
            sorted_patterns = sorted(
                patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            discriminative_patterns[class_name] = []
            for window_key, score in sorted_patterns:
                pattern_info = {
                    'window': window_key,
                    'score': score,
                    'changes': self.patterns_cache.get(window_key, [])
                }
                discriminative_patterns[class_name].append(pattern_info)
                
        return discriminative_patterns
        
    def generate_pattern_report(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        window_size: int = 3,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> str:
        """
        Генерирует текстовый отчет о паттернах
        
        Args:
            data: Тензор формы (batch_size, sequence_length, features)
            labels: Метки классов
            window_size: Размер окна для анализа
            top_k: Количество топ паттернов для каждого класса
            threshold: Порог для определения значимых изменений
            
        Returns:
            Строка с отчетом
        """
        patterns = self.get_discriminative_patterns(data, labels, window_size, top_k, threshold)
        report = []
        
        for class_name, class_patterns in patterns.items():
            report.append(f"\nКласс: {class_name}")
            report.append("-" * 50)
            
            if not class_patterns:
                report.append("\nЗначимых паттернов не найдено")
                continue
                
            for idx, pattern in enumerate(class_patterns, 1):
                report.append(f"\nПаттерн {idx} (score: {pattern['score']:.3f})")
                report.append(f"Временное окно: {pattern['window']}")
                
                if pattern['changes']:
                    report.append("\nИзменения:")
                    for change in pattern['changes']:
                        direction = "повышение" if change['change'] > 0 else "понижение"
                        report.append(
                            f"- {change['feature']}: {direction} "
                            f"(значимость: {change['significance']:.3f})"
                        )
                        
        return "\n".join(report) 