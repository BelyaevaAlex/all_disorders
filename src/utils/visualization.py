import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import torch
import os
import sys
import pandas as pd

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.stone import STONEModel

def plot_spider_channels(
    model: STONEModel,
    class_idx: int,
    channel_names: List[str],
    num_freq_bands: int = 7,
    save_path: Optional[str] = None
) -> None:
    """
    Создает spider plot для визуализации важности каналов.
    
    Args:
        model (STONEModel): Обученная STONE модель
        class_idx (int): Индекс класса
        channel_names (List[str]): Имена каналов
        num_freq_bands (int): Количество частотных диапазонов
        save_path (str, optional): Путь для сохранения графика
    """
    formula = model.formulas[class_idx]
    num_channels = len(channel_names)
    
    # Получаем веса для каждого канала
    weights = []
    for atomic in formula.atomic_formulas:
        # Преобразуем веса в матрицу (каналы, частотные диапазоны + LL)
        w = atomic.a.detach().cpu().numpy()
        # Разделяем веса на частотные диапазоны и LL
        w_freq = w[:num_channels * num_freq_bands].reshape(num_channels, num_freq_bands)
        w_ll = w[num_channels * num_freq_bands:].reshape(num_channels, 1)
        # Усредняем веса по частотным диапазонам и LL
        channel_weights = np.concatenate([w_freq, w_ll], axis=1).mean(axis=1)
        weights.append(channel_weights)
    
    # Усредняем веса по всем атомарным формулам
    avg_weights = np.mean(weights, axis=0)
    
    # Нормализуем веса
    normalized_weights = avg_weights / np.max(avg_weights)
    
    # Создаем spider plot
    angles = np.linspace(0, 2*np.pi, len(channel_names), endpoint=False)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Строим график
    ax.plot(angles, normalized_weights)
    ax.fill(angles, normalized_weights, alpha=0.25)
    
    # Настраиваем оси
    ax.set_xticks(angles)
    ax.set_xticklabels(channel_names)
    ax.set_ylim(0, 1)
    
    plt.title(f'Важность каналов для класса {class_idx}')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_violation_map(
    violation_map: torch.Tensor,
    timestamps: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Визуализирует карту нарушений формулы.
    
    Args:
        violation_map (torch.Tensor): Карта нарушений формы (batch_size, sequence_length)
        timestamps (List[float]): Временные метки
        save_path (str, optional): Путь для сохранения графика
    """
    plt.figure(figsize=(15, 5))
    
    # Преобразуем в numpy
    violation_map = violation_map.detach().cpu().numpy()
    
    # Создаем heatmap
    sns.heatmap(
        violation_map,
        cmap='RdYlGn',
        xticklabels=[f'{t:.1f}' for t in timestamps],
        yticklabels=False,
        cbar_kws={'label': 'Степень удовлетворения формулы'}
    )
    
    plt.xlabel('Время (с)')
    plt.ylabel('Примеры')
    plt.title('Карта нарушений формулы')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_decision_evolution(
    evolution: torch.Tensor,
    timestamps: List[float],
    class_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Визуализирует эволюцию решений во времени.
    
    Args:
        evolution (torch.Tensor): Эволюция решений формы (batch_size, sequence_length, num_classes)
        timestamps (List[float]): Временные метки
        class_names (List[str]): Имена классов
        save_path (str, optional): Путь для сохранения графика
    """
    plt.figure(figsize=(15, 5))
    
    # Преобразуем в numpy
    evolution = evolution.detach().cpu().numpy()
    
    # Усредняем по батчу
    mean_evolution = evolution.mean(axis=0)
    
    # Строим график для каждого класса
    for i in range(mean_evolution.shape[1]):
        plt.plot(timestamps, mean_evolution[:, i], label=class_names[i])
    
    plt.xlabel('Время (с)')
    plt.ylabel('Вероятность')
    plt.title('Эволюция решений во времени')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(
    model: STONEModel,
    class_idx: int,
    feature_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Визуализирует важность признаков для заданного класса.
    
    Args:
        model (STONEModel): Обученная STONE модель
        class_idx (int): Индекс класса
        feature_names (List[str]): Имена признаков
        save_path (str, optional): Путь для сохранения графика
    """
    formula = model.formulas[class_idx]
    
    # Получаем веса для всех атомарных формул
    weights = []
    for atomic in formula.atomic_formulas:
        w = atomic.a.detach().cpu().numpy()
        weights.append(np.abs(w))
    
    # Усредняем веса по всем атомарным формулам
    avg_weights = np.mean(weights, axis=0)
    
    # Создаем DataFrame для удобной работы с данными
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_weights
    })
    
    # Разделяем имя признака на канал и тип
    df[['channel', 'type']] = df['feature'].str.extract(r'([A-Za-z0-9]+)_(.+)')
    
    # Создаем отдельные графики для спектральных признаков и LL
    plt.figure(figsize=(20, 10))
    
    # 1. График спектральных признаков
    plt.subplot(1, 2, 1)
    spectral_df = df[df['type'] != 'LL'].copy()
    spectral_pivot = spectral_df.pivot(
        index='channel',
        columns='type',
        values='importance'
    )
    
    sns.heatmap(
        spectral_pivot,
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Важность признака'}
    )
    plt.title('Важность спектральных признаков')
    plt.xlabel('Частотный диапазон')
    plt.ylabel('Канал ЭЭГ')
    
    # 2. График Line Length
    plt.subplot(1, 2, 2)
    ll_df = df[df['type'] == 'LL'].sort_values('importance', ascending=True)
    
    # Создаем горизонтальный bar plot для LL
    bars = plt.barh(ll_df['channel'], ll_df['importance'])
    plt.title('Важность признаков Line Length')
    plt.xlabel('Важность признака')
    plt.ylabel('Канал ЭЭГ')
    
    # Добавляем значения на бары
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height()/2,
            f'{width:.2f}',
            ha='left',
            va='center'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Дополнительно сохраняем топ-5 признаков каждого типа в текстовый файл
    if save_path:
        txt_path = save_path.rsplit('.', 1)[0] + '_top_features.txt'
        with open(txt_path, 'w') as f:
            # Топ-5 спектральных признаков
            f.write("Топ-5 спектральных признаков:\n")
            top_spectral = df[df['type'] != 'LL'].nlargest(5, 'importance')
            for _, row in top_spectral.iterrows():
                f.write(f"{row['feature']}: {row['importance']:.3f}\n")
            
            # Топ-5 признаков Line Length
            f.write("\nТоп-5 признаков Line Length:\n")
            top_ll = df[df['type'] == 'LL'].nlargest(5, 'importance')
            for _, row in top_ll.iterrows():
                f.write(f"{row['feature']}: {row['importance']:.3f}\n")

if __name__ == "__main__":
    # Пример использования
    batch_size, seq_len, input_dim = 32, 10, 144
    num_classes = 4
    
    # Генерируем тестовые данные
    x = torch.randn(batch_size, seq_len, input_dim)
    timestamps = np.linspace(0, 10, seq_len)
    
    # Создаем модель
    model = STONEModel(
        input_dim=input_dim,
        num_classes=num_classes,
        num_atomic=3,
        eventually_window=(0, 5),
        always_window=(0, 3)
    )
    
    # Получаем карту нарушений
    violation_map = model.get_violation_map(x, class_idx=0)
    plot_violation_map(violation_map, timestamps)
    
    # Получаем эволюцию решений
    evolution = model.get_decision_evolution(x)
    plot_decision_evolution(
        evolution,
        timestamps,
        class_names=[f'Класс {i}' for i in range(num_classes)]
    )
    
    # Визуализируем важность каналов
    channel_names = [f'Channel {i}' for i in range(18)]  # 18 каналов
    plot_spider_channels(model, class_idx=0, channel_names=channel_names)
    
    # Визуализируем важность признаков
    feature_names = [f'Feature {i}' for i in range(input_dim)]
    plot_feature_importance(model, class_idx=0, feature_names=feature_names) 