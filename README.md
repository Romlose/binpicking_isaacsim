# Установка и тестирование
Проект работает для Isaac Sim 5.1.0, для других версий может запуститься но придется некоторые импорты переделывать
1. Скачайте zip и разархивируйте (Windows, Linux без разницы) в папку ~/isaac_sim или C:/User/your_username/isaac_sim
2. Склоньте проект в корень
3. Проект состоит из User extention и основного проекта. Extention нужна для отладки и пока в разработке
4. Запустите проверку совместимости и разогревочный скрипт для создания кеша, шейдеров и др:
```bash
# В корне isaac_sim 
./isaac-sim.compatibility_check.sh
./warmup.sh
```
5. Проверьте работу RGB-D сенсора:
```bash
./python.sh standalone_examples/api/isaacsim.sensors.camera/camera_annotator_device.py
./python.sh standalone_examples/benchmarks/benchmark_camera.py
```

Запуск в debug_mode (рекомендуется, но можно без него)
```bash
./python.sh binpicking_isaacsim/dimplom_franka_3d/main.py -d
```

## Результаты
![Результат симуляции](images/1.png)

# Теория