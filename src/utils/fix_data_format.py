import pandas as pd
import numpy as np
import ast
from typing import Union, List
import os
import json

def is_array_string(value: str) -> bool:
    """Проверяет, является ли строка представлением массива."""
    try:
        if not isinstance(value, str):
            return False
        value = value.strip()
        return (value.startswith('[') and value.endswith(']')) or (value.startswith('"[') and value.endswith(']"'))
    except:
        return False

def clean_array_string(value: str) -> str:
    """Очищает строковое представление массива от лишних кавычек и пробелов."""
    try:
        value = value.strip()
        if value.startswith('"['):
            value = value[1:]
        if value.endswith(']"'):
            value = value[:-1]
        return value
    except:
        return value

def convert_to_array_string(value: Union[str, float, int]) -> str:
    """Преобразует различные форматы данных в строковое представление массива."""
    try:
        if isinstance(value, (float, int)):
            # Если это число, создаем массив из 30250 элементов (как в других каналах)
            array = np.full(30250, float(value))
            return json.dumps(array.tolist())
        elif is_array_string(str(value)):
            # Если это уже строка с массивом, очищаем и проверяем
            clean_value = clean_array_string(str(value))
            try:
                array = ast.literal_eval(clean_value)
                if isinstance(array, list):
                    if len(array) != 30250:
                        # Если длина массива неправильная, создаем новый
                        print(f"Warning: Array length {len(array)} != 30250, padding with zeros")
                        new_array = np.zeros(30250)
                        new_array[:min(len(array), 30250)] = array[:min(len(array), 30250)]
                        return json.dumps(new_array.tolist())
                    return json.dumps(array)
            except:
                # Если не удалось разобрать через ast, пробуем через json
                try:
                    array = json.loads(clean_value)
                    if isinstance(array, list):
                        if len(array) != 30250:
                            new_array = np.zeros(30250)
                            new_array[:min(len(array), 30250)] = array[:min(len(array), 30250)]
                            return json.dumps(new_array.tolist())
                        return json.dumps(array)
                except:
                    raise ValueError(f"Invalid array format: {value}")
        
        raise ValueError(f"Unexpected data format: {value}")
    except Exception as e:
        print(f"Error converting value: {value}, Error: {str(e)}")
        # В случае ошибки возвращаем массив нулей
        return json.dumps(np.zeros(30250).tolist())

def main():
    # Загружаем данные
    csv_path = "/home/belyaeva.a/df_open.csv"
    df = pd.read_csv(csv_path)
    
    # Определяем каналы ЭЭГ
    eeg_channels = [col for col in df.columns if col not in [
        'duration_sec', 'event', 'file_path', 'label',
        'n_channels', 'patient_id', 'sfreq', 'Status'
    ]]
    
    print("Проверяем и исправляем формат данных...")
    
    # Создаем копию для сохранения исправленных данных
    df_fixed = df.copy()
    
    # Проходим по всем каналам и записям
    for channel in eeg_channels:
        print(f"Обрабатываем канал: {channel}")
        problematic_rows = []
        
        for idx in range(len(df)):
            value = df.loc[idx, channel]
            if not is_array_string(str(value)) or len(ast.literal_eval(clean_array_string(str(value)))) != 30250:
                problematic_rows.append(idx)
                df_fixed.loc[idx, channel] = convert_to_array_string(value)
        
        if problematic_rows:
            print(f"Исправлено {len(problematic_rows)} проблемных записей в канале {channel}")
    
    # Сохраняем исправленные данные
    output_path = "/home/belyaeva.a/df_open_fixed.csv"
    df_fixed.to_csv(output_path, index=False)
    print(f"\nГотово! Исправленные данные сохранены в: {output_path}")
    
    # Выводим статистику по размерам массивов
    print("\nПроверяем размеры массивов в исправленных данных...")
    for channel in eeg_channels:
        sample_value = df_fixed.loc[0, channel]
        try:
            array = json.loads(clean_array_string(str(sample_value)))
            print(f"{channel}: {len(array)} точек")
        except Exception as e:
            print(f"{channel}: ошибка при проверке длины массива - {str(e)}")

if __name__ == "__main__":
    main() 