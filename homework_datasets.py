import torch
import pandas as pd
from torch.utils.data import Dataset


### 2.1 Кастомный Dataset класс (15 баллов)
# Создайте кастомный класс датасета для работы с CSV файлами:
# - Загрузка данных из файла
# - Предобработка (нормализация, кодирование категорий)
# - Поддержка различных форматов данных (категориальные, числовые, бинарные и т.д.)

### 2.2 Эксперименты с различными датасетами (15 баллов)
# Найдите csv датасеты для регрессии и бинарной классификации и, применяя наработки из предыдущей части задания, обучите линейную и логистическую регрессию

class PokemonDataset(Dataset):
    def __init__(self, file_path, target='Total'):
        self.df = pd.read_csv(file_path)
        self.df = self.df.dropna().reset_index(drop=True)

        # Кодирование булевых Legendary
        self.df['Legendary'] = self.df['Legendary'].map({True: 1, False: 0, 'True': 1, 'False': 0}).astype(float)

        # Категориальные: Type 1, Type 2
        cat_cols = ['Type 1', 'Type 2']
        dummies = pd.get_dummies(self.df[cat_cols], prefix=['T1', 'T2'], drop_first=True)
        self.df = pd.concat([self.df.drop(columns=cat_cols), dummies], axis=1)

        # Числовые колонки (кроме target)
        num_cols = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation']
        self.scalers = {}
        for col in num_cols:
            mean = self.df[col].mean()
            std = self.df[col].std()
            self.scalers[col] = (mean, std)
            self.df[col] = (self.df[col] - mean) / (std + 1e-8)

        # Подготовка X и y
        feature_cols = [c for c in self.df.columns if c not in ['#','Name', target]]
        # Приводим все к числам (исключаем возможные object)
        data_array = self.df[feature_cols].apply(pd.to_numeric, errors='raise').values.astype(float)
        self.X = torch.tensor(data_array, dtype=torch.float32)
        self.y = torch.tensor(self.df[target].values.astype(float), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MushroomDataset(Dataset):
    def __init__(self, file_path, target_col='class'):
        df = pd.read_csv(file_path)
        df = df.dropna().reset_index(drop=True)

        # Сохраняем список оригинальных классов
        self.classes = df[target_col].astype('category').cat.categories.tolist()
        self.y = torch.tensor(
            df[target_col].astype('category').cat.codes.values,
            dtype=torch.long
        )

        df = df.drop(columns=[target_col])
        df = pd.get_dummies(df, drop_first=True)

        # Приводим к tensor
        self.X = torch.tensor(df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]