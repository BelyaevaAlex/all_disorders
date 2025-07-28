import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np

class STLFormula(nn.Module):
    """
    Базовый класс для STL формул.
    Реализует дифференцируемую версию логических операторов.
    """
    def __init__(self):
        super().__init__()
        
    def _and(self, x: torch.Tensor, y: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
        """Дифференцируемый AND оператор"""
        return torch.min(x, y)
    
    def _or(self, x: torch.Tensor, y: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
        """Дифференцируемый OR оператор"""
        return torch.max(x, y)
    
    def _not(self, x: torch.Tensor) -> torch.Tensor:
        """Дифференцируемый NOT оператор"""
        return 1 - x
    
    def _eventually(self, x: torch.Tensor, window: Tuple[int, int]) -> torch.Tensor:
        """
        Оператор Eventually (◊): истинно, если формула верна хотя бы в один момент
        времени в заданном окне.
        """
        start, end = window
        values = []
        for i in range(start, end + 1):
            if i < x.shape[1]:
                values.append(x[:, i:i+1])
        if not values:
            return torch.zeros_like(x[:, 0:1])
        return torch.max(torch.cat(values, dim=1), dim=1, keepdim=True)[0]
    
    def _always(self, x: torch.Tensor, window: Tuple[int, int]) -> torch.Tensor:
        """
        Оператор Always (□): истинно, если формула верна во все моменты
        времени в заданном окне.
        """
        start, end = window
        values = []
        for i in range(start, end + 1):
            if i < x.shape[1]:
                values.append(x[:, i:i+1])
        if not values:
            return torch.ones_like(x[:, 0:1])
        return torch.min(torch.cat(values, dim=1), dim=1, keepdim=True)[0]

class AtomicFormula(STLFormula):
    """
    Атомарная формула вида a^T x > c
    """
    def __init__(
        self,
        input_dim: int,
        beta: float = 10.0
    ):
        super().__init__()
        self.a = nn.Parameter(torch.randn(input_dim))
        self.c = nn.Parameter(torch.zeros(1))
        self.beta = beta
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Входной тензор формы (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Степень истинности формы (batch_size, sequence_length, 1)
        """
        # Вычисляем a^T x
        projection = F.linear(x, self.a.unsqueeze(0))  # (batch_size, sequence_length, 1)
        
        # Применяем сигмоиду для получения гладкой версии >
        return torch.sigmoid(self.beta * (projection - self.c))

class STONEFormula(STLFormula):
    """
    Реализация STONE формулы для одного класса.
    Формула имеет вид: ◊[t1,t2] □[t3,t4] (conjunction of atomic formulas)
    """
    def __init__(
        self,
        input_dim: int,
        num_atomic: int = 3,
        eventually_window: Tuple[int, int] = (0, 5),
        always_window: Tuple[int, int] = (0, 3),
        beta: float = 10.0
    ):
        """
        Args:
            input_dim (int): Размерность входных данных
            num_atomic (int): Количество атомарных формул в конъюнкции
            eventually_window (Tuple[int, int]): Окно для оператора Eventually
            always_window (Tuple[int, int]): Окно для оператора Always
            beta (float): Параметр сглаживания для логических операторов
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_atomic = num_atomic
        self.eventually_window = eventually_window
        self.always_window = always_window
        self.beta = beta
        
        # Создаем атомарные формулы
        self.atomic_formulas = nn.ModuleList([
            AtomicFormula(input_dim, beta) for _ in range(num_atomic)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Входной тензор формы (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Степень истинности формулы формы (batch_size, 1)
        """
        # Вычисляем значения всех атомарных формул
        atomic_values = []
        for formula in self.atomic_formulas:
            atomic_values.append(formula(x))
        
        # Объединяем атомарные формулы через конъюнкцию
        conjunction = atomic_values[0]
        for value in atomic_values[1:]:
            conjunction = self._and(conjunction, value)
        
        # Применяем темпоральные операторы
        always_result = self._always(conjunction, self.always_window)
        eventually_result = self._eventually(always_result, self.eventually_window)
        
        return eventually_result
    
    def get_formula_string(self, feature_names: Optional[List[str]] = None) -> str:
        """
        Возвращает строковое представление формулы.
        
        Args:
            feature_names (List[str], optional): Имена признаков
            
        Returns:
            str: Строковое представление формулы
        """
        if feature_names is None:
            feature_names = [f"x_{i}" for i in range(self.input_dim)]
            
        formula_parts = []
        for i, atomic in enumerate(self.atomic_formulas):
            # Получаем коэффициенты и порог
            a = atomic.a.detach().cpu().numpy()
            c = atomic.c.item()
            
            # Формируем строку для линейной комбинации
            terms = []
            for j, (coef, name) in enumerate(zip(a, feature_names)):
                if abs(coef) > 1e-3:  # Игнорируем близкие к нулю коэффициенты
                    terms.append(f"{coef:.3f}*{name}")
            
            if terms:
                formula_parts.append(f"({' + '.join(terms)} > {c:.3f})")
        
        # Собираем полную формулу
        conjunction = " ∧ ".join(formula_parts)
        t1, t2 = self.eventually_window
        t3, t4 = self.always_window
        
        return f"◊[{t1},{t2}] □[{t3},{t4}]({conjunction})"

    def get_formula_explanation(self, feature_names: Optional[List[str]] = None) -> str:
        """
        Возвращает понятное объяснение формулы на английском языке.
        
        Args:
            feature_names (List[str], optional): Имена признаков
            
        Returns:
            str: Объяснение формулы на английском языке
        """
        if feature_names is None:
            feature_names = [f"x_{i}" for i in range(self.input_dim)]
            
        # Получаем атомарные формулы
        atomic_explanations = []
        for i, atomic in enumerate(self.atomic_formulas):
            # Получаем коэффициенты и порог
            a = atomic.a.detach().cpu().numpy()
            c = atomic.c.item()
            
            # Находим наиболее значимые признаки (с наибольшими по модулю коэффициентами)
            significant_features = []
            for j, (coef, name) in enumerate(zip(a, feature_names)):
                if abs(coef) > 1e-3:  # Игнорируем близкие к нулю коэффициенты
                    significant_features.append((abs(coef), coef, name))
            
            if significant_features:
                # Сортируем по значимости
                significant_features.sort(reverse=True)
                
                # Берем топ-3 самых значимых признака
                top_features = significant_features[:3]
                
                # Формируем описание
                feature_desc = []
                for _, coef, name in top_features:
                    if coef > 0:
                        feature_desc.append(f"increased activity in {name}")
                    else:
                        feature_desc.append(f"decreased activity in {name}")
                
                if len(feature_desc) > 1:
                    feature_desc[-1] = f"and {feature_desc[-1]}"
                atomic_explanations.append(", ".join(feature_desc))
        
        # Формируем полное объяснение
        t1, t2 = self.eventually_window
        t3, t4 = self.always_window
        
        explanation = "At some point between time steps {} and {}, ".format(t1, t2)
        explanation += "there is a period from {} to {} consecutive time steps ".format(t3, t4)
        explanation += "where we observe {}".format(" together with ".join(atomic_explanations))
        
        return explanation

if __name__ == "__main__":
    # Пример использования
    batch_size, seq_len, input_dim = 32, 10, 144
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Создаем формулу
    formula = STONEFormula(
        input_dim=input_dim,
        num_atomic=3,
        eventually_window=(0, 5),
        always_window=(0, 3)
    )
    
    # Вычисляем значение формулы
    result = formula(x)
    print(f"Размер выхода: {result.shape}")
    
    # Выводим строковое представление формулы
    print("\nФормула:")
    print(formula.get_formula_string()) 