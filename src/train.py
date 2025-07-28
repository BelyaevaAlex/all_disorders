import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os
import json
from typing import Dict, List, Tuple, Optional
import sys
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.stone import STONEModel
from src.utils.data_loader import EEGDataLoader
from src.utils.visualization import (
    plot_spider_channels,
    plot_violation_map,
    plot_decision_evolution,
    plot_feature_importance
)

class WeightedCrossEntropyLoss(nn.Module):
    """
    Взвешенная кросс-энтропия для несбалансированных классов.
    """
    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.class_weights = class_weights
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Отладочная печать размерностей
        print(f"y_pred shape: {y_pred.shape}")
        print(f"y_true shape: {y_true.shape}")
        print(f"class_weights shape: {self.class_weights.shape}")
        
        # Убираем лишнюю размерность из предсказаний
        if y_pred.dim() == 3:
            y_pred = y_pred.squeeze(-1)
        
        # Применяем логарифм к предсказаниям
        log_probs = torch.log(y_pred + 1e-7)
        
        # Создаем one-hot вектор для истинных меток
        num_classes = y_pred.size(-1)
        y_true = y_true.view(-1)  # Преобразуем в одномерный тензор
        y_true_one_hot = torch.zeros(y_true.size(0), num_classes, device=y_pred.device)
        y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)
        
        # Применяем веса классов к one-hot вектору
        weighted_y_true = y_true_one_hot * self.class_weights
        
        # Вычисляем взвешенную кросс-энтропию
        weighted_loss = -(weighted_y_true * log_probs).sum(dim=1)
        
        return weighted_loss.mean()

def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Вычисляет веса классов на основе их частоты.
    """
    class_counts = np.bincount(y)
    total_samples = len(y)
    weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)

def aggregate_predictions_by_record(outputs: torch.Tensor, record_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Агрегирует предсказания по записям.
    
    Args:
        outputs (torch.Tensor): Тензор предсказаний модели [N, num_classes, 1] или [N, num_classes]
        record_ids (torch.Tensor): Тензор ID записей [N]
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Агрегированные предсказания [M, num_classes]
            - Уникальные ID записей [M]
    """
    # Убираем лишнюю размерность из outputs, если она есть
    if outputs.dim() == 3:
        outputs = outputs.squeeze(-1)  # [N, num_classes]
    
    unique_ids = torch.unique(record_ids)
    aggregated_outputs = torch.zeros((len(unique_ids), outputs.size(1)), device=outputs.device)
    
    for i, idx in enumerate(unique_ids):
        mask = (record_ids == idx)
        # Усредняем предсказания для каждой записи
        avg_logits = outputs[mask].mean(dim=0)  # [num_classes]
        # Применяем softmax к усредненным предсказаниям
        # Добавляем размерность batch и применяем softmax
        aggregated_outputs[i] = F.softmax(avg_logits.unsqueeze(0), dim=0)
    
    return aggregated_outputs, unique_ids

def calculate_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    record_ids: torch.Tensor,
    num_classes: int
) -> Dict:
    """
    Рассчитывает подробные метрики с учетом агрегации по окнам.
    
    Args:
        outputs (torch.Tensor): Выходы модели (после softmax)
        targets (torch.Tensor): Истинные метки
        record_ids (torch.Tensor): ID записей для каждого окна
        num_classes (int): Количество классов
        
    Returns:
        Dict: Словарь с метриками
    """
    # Агрегируем предсказания по записям
    aggregated_outputs, unique_ids = aggregate_predictions_by_record(outputs, record_ids)
    
    # Получаем метки для уникальных записей
    unique_targets = torch.zeros(len(unique_ids), dtype=torch.long, device=outputs.device)
    for idx, record_id in enumerate(unique_ids):
        unique_targets[idx] = targets[record_ids == record_id][0]  # Берем метку первого окна
    
    # Получаем предсказания
    predicted = aggregated_outputs.argmax(dim=1)
    
    # Базовые метрики
    correct = (predicted == unique_targets).sum().item()
    total = len(unique_targets)
    accuracy = 100.0 * correct / total
    
    # Метрики по классам
    class_correct = torch.zeros(num_classes, device=outputs.device)
    class_total = torch.zeros(num_classes, device=outputs.device)
    class_precision = torch.zeros(num_classes, device=outputs.device)
    
    for i in range(num_classes):
        # True Positives: правильно предсказанные примеры класса i
        mask = unique_targets == i
        class_total[i] = mask.sum().item()
        class_correct[i] = ((predicted == i) & mask).sum().item()
        
        # Precision: доля правильных предсказаний среди всех предсказаний класса i
        predicted_as_i = (predicted == i).sum().item()
        class_precision[i] = class_correct[i] / predicted_as_i if predicted_as_i > 0 else 0
    
    # Средние метрики
    class_accuracy = 100.0 * class_correct / class_total.clamp(min=1)
    mean_class_accuracy = class_accuracy.mean().item()
    mean_precision = class_precision.mean().item()
    
    return {
        'accuracy': accuracy,
        'mean_class_accuracy': mean_class_accuracy,
        'mean_precision': mean_precision,
        'class_accuracy': class_accuracy.cpu().numpy(),
        'class_precision': class_precision.cpu().numpy(),
        'confusion': {
            'predicted': predicted.cpu().numpy(),
            'true': unique_targets.cpu().numpy()
        }
    }

def train_model(
    model: STONEModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    num_epochs: int,
    device: torch.device,
    save_dir: str,
    early_stopping_patience: int = 10,
    label_mapping: Dict = None
) -> Tuple[List[float], List[float]]:
    """
    Обучает модель и возвращает историю потерь.
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # История метрик
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0
        epoch_train_metrics = []
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        
        for batch_x, batch_y, batch_record_ids in train_progress:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_record_ids = batch_record_ids.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Градиентный клиппинг
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Считаем метрики для батча с учетом агрегации по окнам
            batch_metrics = calculate_metrics(outputs, batch_y, batch_record_ids, model.num_classes)
            epoch_train_metrics.append(batch_metrics)
            
            # Обновляем прогресс-бар
            train_progress.set_postfix({
                'loss': f'{loss.item():.2f}',
                'acc': f'{batch_metrics["accuracy"]:.2f}%',
                'mean_cls_acc': f'{batch_metrics["mean_class_accuracy"]:.2f}%'
            })
        
        # Усредняем метрики по эпохе
        avg_train_loss = train_loss / len(train_loader)
        avg_train_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in epoch_train_metrics]),
            'mean_class_accuracy': np.mean([m['mean_class_accuracy'] for m in epoch_train_metrics]),
            'mean_precision': np.mean([m['mean_precision'] for m in epoch_train_metrics])
        }
        
        # Валидация
        model.eval()
        val_loss = 0
        epoch_val_metrics = []
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for batch_x, batch_y, batch_record_ids in val_progress:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_record_ids = batch_record_ids.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                # Считаем метрики для батча
                batch_metrics = calculate_metrics(outputs, batch_y, batch_record_ids, model.num_classes)
                epoch_val_metrics.append(batch_metrics)
                
                # Обновляем прогресс-бар
                val_progress.set_postfix({
                    'loss': f'{loss.item():.2f}',
                    'acc': f'{batch_metrics["accuracy"]:.2f}%',
                    'mean_cls_acc': f'{batch_metrics["mean_class_accuracy"]:.2f}%'
                })
        
        # Усредняем метрики по эпохе
        avg_val_loss = val_loss / len(val_loader)
        avg_val_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in epoch_val_metrics]),
            'mean_class_accuracy': np.mean([m['mean_class_accuracy'] for m in epoch_val_metrics]),
            'mean_precision': np.mean([m['mean_precision'] for m in epoch_val_metrics])
        }
        
        # Обновляем learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Сохраняем метрики в историю
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_metrics'].append(avg_train_metrics)
        history['val_metrics'].append(avg_val_metrics)
        
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Train Accuracy: {avg_train_metrics["accuracy"]:.2f}%')
        print(f'Train Mean Class Accuracy: {avg_train_metrics["mean_class_accuracy"]:.2f}%')
        print(f'Train Mean Precision: {avg_train_metrics["mean_precision"]:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Accuracy: {avg_val_metrics["accuracy"]:.2f}%')
        print(f'Val Mean Class Accuracy: {avg_val_metrics["mean_class_accuracy"]:.2f}%')
        print(f'Val Mean Precision: {avg_val_metrics["mean_precision"]:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Сохраняем лучшую модель
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'metrics': avg_val_metrics
            }, os.path.join(save_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break
    
    return history

def parse_args():
    parser = argparse.ArgumentParser(description='Обучение модели STONE')
    
    # Параметры данных
    parser.add_argument('--data_path', type=str, required=True, help='Путь к файлу данных')
    parser.add_argument('--num_rows', type=int, default=None, help='Количество строк для чтения из CSV (по умолчанию: все строки)')
    parser.add_argument('--window_size', type=float, default=10.0, help='Размер окна в секундах')
    parser.add_argument('--step_size', type=float, default=0.2, help='Шаг окна в секундах')
    
    # Параметры обучения
    parser.add_argument('--num_epochs', type=int, default=30, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Скорость обучения')
    
    # Параметры модели STONE
    parser.add_argument('--num_atomic', type=int, default=10, help='Количество атомарных формул на класс')
    parser.add_argument('--eventually_window', type=tuple, default=(0,8), help='Окно для оператора Eventually')
    parser.add_argument('--always_window', type=tuple, default=(0,5), help='Окно для оператора Always')
    parser.add_argument('--beta', type=float, default=1.0, help='Параметр beta для сигмоиды')
    
    return parser.parse_args()

def main():
    # Параметры командной строки
    args = parse_args()

    # Параметры модели
    num_atomic = args.num_atomic  # Увеличиваем количество атомарных формул
    eventually_window = args.eventually_window  # Расширяем окно для оператора Eventually
    always_window = args.always_window  # Расширяем окно для оператора Always
    beta = args.beta  # Увеличенное значение beta для более резких границ
    
    # Создаем директорию для результатов
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Загрузка и разделение данных
    print("\n1. Загрузка и разделение данных...")
    
    # Определяем устройство
    torch.cuda.set_device(2)  # Явно указываем использовать GPU 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nИспользуется устройство: {device} (GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name()})")
    
    # Загружаем данные
    data_loader = EEGDataLoader(args.data_path)
    X, y, record_ids, label_mapping = data_loader.load_data()
    
    # Создаем полный датасет
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y),
        torch.LongTensor(record_ids)
    )
    
    # Разделяем данные на train (70%), validation (15%) и test (15%)
    # Сначала отделяем train
    train_size = int(0.7 * len(dataset))
    temp_size = len(dataset) - train_size
    train_dataset, temp_dataset = train_test_split(
        dataset, 
        test_size=temp_size,
        random_state=42,
        stratify=y
    )
    
    # Затем разделяем оставшиеся данные на validation и test поровну
    val_size = temp_size // 2
    test_size = temp_size - val_size
    val_dataset, test_dataset = train_test_split(
        temp_dataset,
        test_size=test_size,
        random_state=42,
        stratify=torch.tensor([y[i] for i in range(len(y)) if i >= train_size])
    )
    
    print(f"Распределение данных:")
    print(f"Train: {len(train_dataset)} окон")
    print(f"Validation: {len(val_dataset)} окон")
    print(f"Test: {len(test_dataset)} окон")
    
    # Вычисляем веса классов используя только тренировочные данные
    train_y = torch.tensor([y for _, y, _ in train_dataset])
    class_counts = torch.bincount(train_y)
    total_samples = len(train_y)
    class_weights = torch.FloatTensor(total_samples / (len(class_counts) * class_counts))
    class_weights = class_weights.to(device)
    
    # Создаем веса для сэмплирования на основе классов
    sample_weights = torch.zeros(len(train_dataset))
    for idx, (_, label, _) in enumerate(train_dataset):
        sample_weights[idx] = class_weights[label]
    
    # Создаем sampler для тренировочного датасета
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True  # Разрешаем повторное использование примеров в рамках эпохи
    )
    
    print("\nРаспределение классов в тренировочном наборе:")
    for i in range(len(class_counts)):
        print(f"Класс {i}: {class_counts[i]} примеров (вес: {class_weights[i]:.4f})")
    
    # Создаем загрузчики данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,  # Используем weighted sampler вместо shuffle
        num_workers=4,  # Добавляем многопоточную загрузку данных
        pin_memory=True  # Оптимизация для GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Создаем модель
    model = STONEModel(
        input_dim=X.shape[2],
        num_classes=len(class_counts),
        num_atomic=args.num_atomic,
        eventually_window=eventually_window,
        always_window=always_window,
        beta=beta
    ).to(device)

    # Создаем оптимизатор и функцию потерь
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01  # L2 регуляризация
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    criterion = WeightedCrossEntropyLoss(class_weights)

    # Обучаем модель
    print("\nНачало обучения...")
    history = train_model(
        model=model,
        train_loader=train_loader,  # Используем правильный train_loader
        val_loader=val_loader,      # Используем правильный val_loader
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=save_dir,
        label_mapping=label_mapping
    )
    
    # Оценка на тестовом наборе
    print("\n4. Оценка на тестовом наборе...")
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc='Testing')  # Используем test_loader
        for batch_x, batch_y, batch_record_ids in test_progress:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_record_ids = batch_record_ids.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            
            # Агрегируем предсказания по записям
            aggregated_outputs, unique_ids = aggregate_predictions_by_record(outputs, batch_record_ids)
            
            # Получаем метки для уникальных записей
            unique_targets = torch.zeros(len(unique_ids), dtype=torch.long, device=outputs.device)
            for idx, record_id in enumerate(unique_ids):
                # Переносим тензоры на CPU перед использованием для индексации
                record_mask = (batch_record_ids == record_id).cpu()
                batch_y_cpu = batch_y.cpu()
                unique_targets[idx] = batch_y_cpu[record_mask][0].to(device)
            
            # Получаем предсказанные классы
            predicted = aggregated_outputs.argmax(dim=1)
            
            # Обновляем счетчики для точности
            test_total += len(unique_ids)
            test_correct += (predicted == unique_targets).sum().item()
            
            # Переносим тензоры на CPU перед конвертацией в numpy
            predicted_cpu = predicted.cpu()
            unique_targets_cpu = unique_targets.cpu()
            
            # Добавляем предсказания и метки для текущего батча
            all_preds.extend(predicted_cpu.numpy().tolist())
            all_true.extend(unique_targets_cpu.numpy().tolist())
            
            # Обновляем прогресс-бар с правильной точностью
            current_accuracy = 100.0 * test_correct / test_total if test_total > 0 else 0.0
            test_progress.set_postfix({
                'loss': f'{loss.item():.2f}',
                'acc': f'{current_accuracy:.2f}%'
            })
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100.0 * test_correct / test_total if test_total > 0 else 0.0
    print(f'\nТестовые метрики:')
    print(f'Loss: {avg_test_loss:.4f}')
    print(f'Accuracy: {test_accuracy:.2f}%')
    
    # Вычисляем и выводим подробные метрики
    print("\nПодробный отчет по классификации:")
    class_names = [k for k, v in sorted(label_mapping.items(), key=lambda x: x[1])]

    # Преобразуем списки в numpy массивы
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    # Отладочная информация
    print(f"\nРазмерности массивов:")
    print(f"Предсказания (all_preds): {all_preds.shape}")
    print(f"Истинные метки (all_true): {all_true.shape}")
    print(f"\nУникальные значения:")
    print(f"Предсказания: {np.unique(all_preds)}")
    print(f"Истинные метки: {np.unique(all_true)}")
    print(f"\nРаспределение классов:")
    print("Предсказания:")
    for i in range(len(class_names)):
        count = np.sum(all_preds == i)
        print(f"{class_names[i]}: {count}")
    print("\nИстинные метки:")
    for i in range(len(class_names)):
        count = np.sum(all_true == i)
        print(f"{class_names[i]}: {count}")

    # Определяем, какие классы присутствуют в данных
    present_classes = sorted(list(set(np.unique(all_true)) | set(np.unique(all_preds))))
    present_class_names = [class_names[i] for i in present_classes]
    
    print("\nОтчет по классификации:")
    if len(present_classes) > 1:
        # Создаем словарь для маппинга старых индексов на новые
        label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(present_classes)}
        
        # Преобразуем метки в новую нумерацию
        mapped_preds = np.array([label_map[p] for p in all_preds])
        mapped_true = np.array([label_map[t] for t in all_true])
        
        print(classification_report(
            mapped_true,
            mapped_preds,
            target_names=present_class_names,
            zero_division=0
        ))
        
        # Сохраняем отчет
        report_dict = classification_report(
            mapped_true,
            mapped_preds,
            target_names=present_class_names,
            zero_division=0,
            output_dict=True
        )
    else:
        accuracy = np.mean(all_true == all_preds)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nПримечание: В данных присутствует только один класс.")
        
        # Создаем отчет в формате словаря
        report_dict = {
            'accuracy': accuracy,
            present_class_names[0]: {
                'precision': 1.0 if accuracy == 1.0 else 0.0,
                'recall': 1.0 if accuracy == 1.0 else 0.0,
                'f1-score': 1.0 if accuracy == 1.0 else 0.0,
                'support': len(all_true)
            }
        }

    # Создаем и сохраняем confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Сохраняем метрики в history
    history['test_loss'] = avg_test_loss
    history['test_accuracy'] = test_accuracy
    history['classification_report'] = report_dict # Сохраняем отчет в формате словаря
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    # 6. Визуализация и интерпретация
    print("\n6. Создание визуализаций...")
    
    # Получаем имена каналов и признаков
    channel_names = data_loader.eeg_channels  # Используем сохраненный EEGDataLoader
    feature_names = []
    for channel in channel_names:
        for band in ['delta', 'theta', 'alpha', 'beta_low', 'beta_high', 'gamma_low', 'gamma_high']:
            feature_names.append(f"{channel}_{band}")
    for channel in channel_names:
        feature_names.append(f"{channel}_LL")
    
    # Создаем визуализации для каждого класса
    for class_idx in range(len(label_mapping)): # Используем len(label_mapping) для количества классов
        class_name = [k for k, v in label_mapping.items() if v == class_idx][0]
        print(f"\nСоздание визуализаций для класса {class_name}...")
        
        # Spider plot каналов
        plot_spider_channels(
            model,
            class_idx,
            channel_names,
            save_path=os.path.join(save_dir, f'spider_plot_class_{class_idx}.png')
        )
        
        # Важность признаков
        plot_feature_importance(
            model,
            class_idx,
            feature_names,
            save_path=os.path.join(save_dir, f'feature_importance_class_{class_idx}.png')
        )
    
    # Визуализируем карту нарушений и эволюцию решений для одного батча
    batch_x, batch_y, batch_record_ids = next(iter(test_loader))
    batch_x = batch_x.to(device)
    batch_record_ids = batch_record_ids.to(device)
    timestamps = np.linspace(0, args.window_size, batch_x.shape[1])
    
    print("\nСоздание карт нарушений и эволюции решений...")
    for class_idx in range(len(label_mapping)): # Используем len(label_mapping) для количества классов
        class_name = [k for k, v in label_mapping.items() if v == class_idx][0]
        violation_map = model.get_violation_map(batch_x, class_idx)
        plot_violation_map(
            violation_map,
            timestamps,
            save_path=os.path.join(save_dir, f'violation_map_class_{class_idx}.png')
        )
    
    evolution = model.get_decision_evolution(batch_x)
    plot_decision_evolution(
        evolution,
        timestamps,
        [k for k, v in sorted(label_mapping.items(), key=lambda x: x[1])],
        save_path=os.path.join(save_dir, 'decision_evolution.png')
    )
    
    print("\nОбучение и анализ завершены. Результаты сохранены в директории:", save_dir)

    # Выводим изученные формулы и их объяснения
    print("\nИзученные wSTL формулы для каждого класса:")
    formulas = model.get_formulas_string(feature_names)
    for class_idx, (class_name, formula) in enumerate(zip(class_names, formulas)):
        print(f"\nКласс: {class_name}")
        print(f"Формула: {formula}")
        print("Объяснение:")
        explanation = model.formulas[class_idx].get_formula_explanation(feature_names)
        print(explanation)

    # Сохраняем формулы и объяснения в файл
    with open(os.path.join(save_dir, 'learned_formulas.txt'), 'w') as f:
        f.write("Learned wSTL Formulas\n")
        f.write("===================\n\n")
        for class_idx, (class_name, formula) in enumerate(zip(class_names, formulas)):
            f.write(f"Class: {class_name}\n")
            f.write(f"Formula: {formula}\n")
            f.write("Explanation:\n")
            explanation = model.formulas[class_idx].get_formula_explanation(feature_names)
            f.write(explanation + "\n\n")

    # Добавляем анализ паттернов
    print("\n7. Анализ паттернов ЭЭГ...")
    from src.utils.pattern_analysis import PatternAnalyzer
    
    # Создаем анализатор паттернов
    pattern_analyzer = PatternAnalyzer(feature_names, class_names)
    
    # Анализируем паттерны на тестовом наборе
    print("Анализируем паттерны на тестовом наборе...")
    # X_test, y_test, record_ids_test = test_data_loader.load_data() # Этот код уже выполнен выше
    test_patterns = pattern_analyzer.get_discriminative_patterns(
        X, # Используем X, y, record_ids из загрузчика данных
        y,
        window_size=3,  # Используем окно в 3 временных шага
        top_k=5  # Топ-5 паттернов для каждого класса
    )
    
    # Генерируем отчет
    print("Генерируем отчет о паттернах...")
    pattern_report = pattern_analyzer.generate_pattern_report(
        X, # Используем X, y, record_ids из загрузчика данных
        y,
        window_size=3,
        top_k=5
    )
    
    # Сохраняем результаты анализа паттернов
    patterns_dir = os.path.join(save_dir, 'patterns')
    os.makedirs(patterns_dir, exist_ok=True)
    
    with open(os.path.join(patterns_dir, 'patterns.json'), 'w', encoding='utf-8') as f:
        json.dump(test_patterns, f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(patterns_dir, 'pattern_report.txt'), 'w', encoding='utf-8') as f:
        f.write(pattern_report)
    
    print("\nАнализ паттернов завершен. Результаты сохранены в директории:", patterns_dir)
    print("\nОтчет о паттернах:")
    print(pattern_report)

if __name__ == "__main__":
    main() 