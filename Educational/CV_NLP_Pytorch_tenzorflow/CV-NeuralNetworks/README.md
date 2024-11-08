# Распознавание рукописных символов EMNIST

## 1. Описание решения
_Опишите в этом разделе задание и ваше решение:_
- _тип задачи_
- _опишите данные и приведите примеры_
- _какую модель вы выбрали_
- _перечислите гиперпараметры_
- _каковы метрики вашей модели на тестовых данных_

В задании была поставлена задача классификации. Тренировочные данные представляли собой 112 800 
сэймплов, каждый из которых был матрицей 28х28.Тестовые данные представляли собой 18800 
сэймплов, каждый из которых был матрицей 28х28. Каждой матрице соотвтствовал лэйбл с 
зашфрованным символом. Расшифровка осуществляется функцией converting. Была выбрана нейросеть 
на базе PyTorch со следующими характеристиками:
MMM(
  (layer1): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (layer2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (layer3): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc1): Linear(in_features=512, out_features=6000, bias=True)
  (drop): Dropout2d(p=0.25, inplace=False)
  (fc2): Linear(in_features=6000, out_features=1200, bias=True)
  (fc3): Linear(in_features=1200, out_features=400, bias=True)
  (fc4): Linear(in_features=400, out_features=47, bias=True)
)
Показатель accuracy равен 0,877, что выше порога 0,87.


## 2. Установка и запуск сервиса

_Опишите в этом разделе, как запустить ваше решение, где должен запуститься сервис, как им пользоваться. Если вы хотите сообщить пользователям и проверяющим дополнительную информацию, сделайте это здесь._

```bash
git clone git@gitlab.skillbox.ru:mikhail_astoshonok/ml-advanced.git
cd ml-advanced/CV-NeuralNetworks
docker build -t cnn6 .
docker run -p 8000:8000 cnn6
```
