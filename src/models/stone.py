import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from .formula import STONEFormula

class STONEModel(nn.Module):
    """
    Мультиклассовая STONE модель.
    Для каждого класса создается отдельная STONE формула.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_atomic: int = 3,
        eventually_window: Tuple[int, int] = (0, 5),
        always_window: Tuple[int, int] = (0, 3),
        beta: float = 10.0
    ):
        """
        Args:
            input_dim (int): Размерность входных данных
            num_classes (int): Количество классов
            num_atomic (int): Количество атомарных формул в каждой STONE формуле
            eventually_window (Tuple[int, int]): Окно для оператора Eventually
            always_window (Tuple[int, int]): Окно для оператора Always
            beta (float): Параметр сглаживания для логических операторов
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Создаем формулу для каждого класса
        self.formulas = nn.ModuleList([
            STONEFormula(
                input_dim=input_dim,
                num_atomic=num_atomic,
                eventually_window=eventually_window,
                always_window=always_window,
                beta=beta
            )
            for _ in range(num_classes)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Входной тензор формы (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Вероятности классов формы (batch_size, num_classes)
        """
        # Вычисляем значения всех формул
        formula_values = []
        for formula in self.formulas:
            value = formula(x)  # (batch_size, 1)
            formula_values.append(value)
        
        # Объединяем значения формул
        logits = torch.cat(formula_values, dim=1)  # (batch_size, num_classes)
        
        # Применяем softmax
        probabilities = F.softmax(logits, dim=1)
        
        return probabilities
    
    def get_formulas_string(self, feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Возвращает строковые представления всех формул.
        
        Args:
            feature_names (List[str], optional): Имена признаков
            
        Returns:
            List[str]: Список строковых представлений формул
        """
        return [formula.get_formula_string(feature_names) for formula in self.formulas]
    
    def get_violation_map(
        self,
        x: torch.Tensor,
        class_idx: int
    ) -> torch.Tensor:
        """
        Вычисляет карту нарушений формулы для заданного класса.
        
        Args:
            x (torch.Tensor): Входной тензор формы (batch_size, sequence_length, input_dim)
            class_idx (int): Индекс класса
            
        Returns:
            torch.Tensor: Карта нарушений формы (batch_size, sequence_length)
        """
        formula = self.formulas[class_idx]
        
        # Вычисляем значения атомарных формул
        atomic_values = []
        for atomic in formula.atomic_formulas:
            value = atomic(x)  # (batch_size, sequence_length, 1)
            atomic_values.append(value)
        
        # Объединяем через конъюнкцию
        conjunction = atomic_values[0]
        for value in atomic_values[1:]:
            conjunction = formula._and(conjunction, value)
        
        # Убираем последнюю размерность
        return conjunction.squeeze(-1)
    
    def get_decision_evolution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Получает эволюцию решений во времени.
        
        Args:
            x (torch.Tensor): Входной тензор формы (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Тензор эволюции решений формы (batch_size, sequence_length, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        evolution = torch.zeros((batch_size, seq_len, self.num_classes), device=x.device)
        
        # Для каждой временной точки
        for t in range(seq_len):
            # Получаем предсказания для текущего временного шага
            with torch.no_grad():
                # Используем только данные до текущего момента времени
                current_x = x[:, :t+1, :]
                outputs = self(current_x)  # (batch_size, num_classes)
                evolution[:, t, :] = outputs.detach().reshape(batch_size, -1)  # Копируем предсказания в соответствующую временную точку
        
        return evolution

if __name__ == "__main__":
    # Пример использования
    batch_size, seq_len, input_dim = 32, 10, 144
    num_classes = 4
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Создаем модель
    model = STONEModel(
        input_dim=input_dim,
        num_classes=num_classes,
        num_atomic=3,
        eventually_window=(0, 5),
        always_window=(0, 3)
    )
    
    # Прямой проход
    probabilities = model(x)
    print(f"Размер выхода: {probabilities.shape}")
    
    # Получаем строковые представления формул
    formulas = model.get_formulas_string()
    print("\nФормулы:")
    for i, formula in enumerate(formulas):
        print(f"Класс {i}: {formula}")
    
    # Получаем карту нарушений для первого класса
    violation_map = model.get_violation_map(x, class_idx=0)
    print(f"\nРазмер карты нарушений: {violation_map.shape}")
    
    # Получаем эволюцию решений
    evolution = model.get_decision_evolution(x)
    print(f"\nРазмер эволюции решений: {evolution.shape}") 