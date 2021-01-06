# Запуск сети на Intel Movidius NCS2

## Содержание
1. [Введение](#intro)
2. [Конфигурация машины](#conf)
3. [Предварительные действия: тренировка и подготовка сети](#preliminaries)
4. [Преобразование в формат IR](#convert)
5. [Загрузка и выполнение сети в NCS2 с использованием python API](#run)

<a name="intro"></a>
## Введение

В данном мануале описывается процедура запуска нейронной сети на ускорителе Intel Movidius Neural 
Compute Stick 2. Сеть предварительно натренирована с использованием библиотеки pytorch. Если кратко, то 
алгоритм запуска сети на ускорителе выглядит следующим образом.
1. Тренировка сети в pytorch и сохранение натренированной модели в формате onnx.
2. Преобразование модели из onnx в IR (intermediate representation). IR - это внутренни формат 
   openvino, в котором хранится оптимизированный граф и веса. В составе openvino идет специальный скрипт
   (model_optimizer), который позволяет преобразовывать чекпоинты из популярных форматов в IR. 
3. Загрузка и выполнение сети в NCS.

Ниже приведено детальное описание этих шагов. Для удобства настройки окружения, воспользуемся заранее 
собранным [docker образом](https://hub.docker.com/r/openvino/ubuntu18_dev) с установленной в нем библиотекой openvino.

<a name="conf"></a>
## Конфигурация машины 

1. ОС Ubuntu 18.04.5
2. Openvino 2021.2
3. Intel Movidius Neural Compute Stick 2

<a name="preliminaries"></a>
## Предварительные действия: тренировка и подготовка сети 

Согласно документации Intel, openvino не поддерживает нативный формат чекпоинтов pytorch и требуется
дополнительное преобразование в формат Onnx.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(4, 50)
        self.lin2 = nn.Linear(50, 50)
        self.out = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.out(x)
        return x
    
# Create and train model here
net = Net()

#  Save the model as ONNX
torch.onnx.export(net, torch.tensor([1, 1, 1, 1], dtype=torch.float), "pytorch_iris.onnx", verbose=True)
```

<a name="convert"></a>
## Преобразование в формат IR

1. Вставить NCS2 в USB
2. Подгрузить и запустить контейнер с openvino
    ```bash
    docker run -it -u 0 --gpus all --device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb -v /:/host --network host --rm --name ncs openvino/ubuntu18_dev:latest
    ```    
    Эта команда подгрузит наиболее актуальный образ ubuntu18_dev и запустит в нем терминал под 
    рутовым пользователем и активирует окружение openvino.
   
3. Активировать NCS2 под докером: 
   ```bash
   cd /opt/intel/openvino_2021.2.185/install_dependencies
   ./install_NCS_udev_rules.sh
   ```
4. Перейти в каталог с сохраненной моделью и вызвать в ней model_optimizer для onnx
   ```bash
   mo_onnx.py --input_model pytorch_iris.onnx
   ```
   В результате должно получиться два файла:
   - pytorch_iris.bin - веса модели
   - pytorch_iris.xml - граф модели
    
<a name="run"></a>
## Загрузка и выполнение сети в NCS2 с использованием python API 

Для запуска сети в NCS2 можно воспользоваться python API из openvino. Пример загрузки и запуска сети
приведен в коде ниже.

- Загрузка сети
    ```python
    from openvino.inference_engine import IENetwork, IECore
    
    
    DEVICE = "MYRIAD"  # "CPU" | "MYRIAD" | "GPU"
    
    
    ie = IECore()
    net = IENetwork(model="pytorch_iris.xml", weights="pytorch_iris.bin")
    
    # Set up the input and output blobs
    gn_input_blob = next(iter(net.inputs))
    gn_output_blob = next(iter(net.outputs))
    gn_input_shape = net.inputs[gn_input_blob].shape
    gn_output_shape = net.outputs[gn_output_blob].shape
    
    # Load the network
    gn_exec_net = ie.load_network(network=net, device_name=DEVICE)
    ```
  
- Выполнение сети на ускорителе
    ```python
    gn_exec_net.infer({gn_input_blob: X})
    ```
