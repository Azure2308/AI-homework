import torch


# 1.1 Создание тензоров
tensor1 = torch.rand(3, 4)
tensor2 = torch.zeros(2, 3, 4)
tensor3 = torch.ones(5, 5)
tensor4 = torch.arange(16).reshape(4, 4)

print("1.1 Создание тензоров (7 баллов):")
print("Тензор размером 3x4, заполненный случайными числами от 0 до 1:\n", tensor1)
print("Тензор размером 2x3x4, заполненный нулями:\n", tensor2)
print("Тензор размером 5x5, заполненный единицами:\n", tensor3)
print("Тензор размером 4x4 с числами от 0 до 15 (используйте reshape):\n", tensor4)
print()

# 1.2 Операции с тензорами
tensorA = torch.randn(3, 4)
tensorB = torch.randn(4, 3)

print("1.2 Операции с тензорами (6 баллов):")
print("A:\n", tensorA)
print("B:\n", tensorB)
print("Транспонирование тензора A:\n", tensorA.T)
print("Матричное умножение A и B:\n", tensorA @ tensorB)
print("Поэлементное умножение A и транспонированного B:\n", tensorA * tensorB.t())
print("Сумма всех элементов тензора A:", tensorA.sum())
print()

# 1.3 Индексация и срезы
tensorC = torch.arange(5*5*5).reshape(5,5,5)

print("1.3 Индексация и срезы (6 баллов):")
print("T shape:", tensorC)
print("Первая строка:\n", tensorC[0, 0, :])
print("Последний столбец:\n", tensorC[:, :, -1] )
print("Подматрица размером 2x2 из центра тензора:\n", tensorC[2, 1:3, 1:3])
print("Все элементы с четными индексами:\n", tensorC[::2, ::2, ::2] )
print()

# 1.4 Работа с формами
tensorD = torch.arange(24)

print("1.4 Работа с формами (6 баллов):")
print("Создайте тензор размером 24 элемента:", tensorD)
print("Форма 2x12:", tensorD.view(2, 12))
print("Форма 3x8:", tensorD.view(3, 8))
print("Форма 4x6:", tensorD.view(4, 6))
print("Форма 2x3x4:", tensorD.reshape(2, 3, 4))
print("Форма 2x2x2x3:", tensorD.reshape(2, 2, 2, 3))
