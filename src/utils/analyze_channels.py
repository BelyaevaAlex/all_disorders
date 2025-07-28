import pandas as pd
import numpy as np

def analyze_channel_values(df, channel_name):
    """Анализирует все уникальные значения в канале."""
    print(f"\nАнализ значений в канале {channel_name}:")
    print("-" * 50)
    
    values = df[channel_name].values
    
    # Анализируем типы данных
    print("Анализ типов данных:")
    value_types = set(type(x) for x in values)
    for t in value_types:
        count = sum(isinstance(x, t) for x in values)
        print(f"Тип {t.__name__}: {count} записей")
    
    # Анализируем числовые значения
    numeric_values = []
    non_numeric_values = []
    
    for idx, val in enumerate(values):
        try:
            if isinstance(val, (int, float)):
                numeric_values.append((idx, val))
            elif isinstance(val, str):
                # Пробуем преобразовать строку в число
                try:
                    num = float(val)
                    numeric_values.append((idx, num))
                except:
                    non_numeric_values.append((idx, val))
        except:
            non_numeric_values.append((idx, val))
    
    print(f"\nВсего записей: {len(values)}")
    print(f"Числовых значений: {len(numeric_values)}")
    print(f"Нечисловых значений: {len(non_numeric_values)}")
    
    if numeric_values:
        print("\nЧисловые значения (первые 10):")
        for idx, val in numeric_values[:10]:
            print(f"Индекс {idx}: {val}")
            
    if non_numeric_values:
        print("\nНечисловые значения (первые 10):")
        for idx, val in non_numeric_values[:10]:
            if isinstance(val, str):
                # Показываем только начало длинных строк
                val_str = val[:200] + "..." if len(val) > 200 else val
            else:
                val_str = str(val)
            print(f"Индекс {idx}: {val_str}")

def main():
    # Загружаем данные
    csv_path = "/home/belyaeva.a/df_open.csv"
    df = pd.read_csv(csv_path)
    
    # Анализируем проблемные каналы
    problem_channels = ['A1', 'A2', 'Fpz']
    
    for channel in problem_channels:
        analyze_channel_values(df, channel)

if __name__ == "__main__":
    main() 