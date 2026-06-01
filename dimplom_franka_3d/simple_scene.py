import sys
from omni.isaac.kit import SimulationApp

# 1. Запуск симуляционного приложения (обязательно до импорта других модулей Isaac)
# False означает, что симуляция запускается с графическим интерфейсом (GUI)
simulation_app = SimulationApp({"headless": False})

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.nucleus import get_assets_root_path

def main():
    # 2. Получаем базовый URL-путь к облачным ассетам на AWS S3 (Omniverse Nucleus)
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        print("Ошибка: Не удалось получить путь к облачным ассетам NVIDIA AWS!")
        simulation_app.close()
        return

    print(f"Успешное подключение к серверу ассетов. Базовый путь: {assets_root_path}")

    # 3. Путь к готовой сцене склада на сервере
    # В Isaac Sim 5.1.0 стандартный склад находится по этому пути:
    warehouse_usd_path = "https://amazonaws.com"

    print(f"Загрузка сцены склада из AWS S3: {warehouse_usd_path}")
    print("Внимание: Первая загрузка может занять несколько минут, так как файлы скачиваются из облака...")

    # 4. Открываем (заменяем текущую сцену) или добавляем склад как под-сцену (Reference)
    # В данном случае мы полностью открываем файл сцены склада
    stage_utils.open_stage(usd_path=warehouse_usd_path)

    # 5. Инициализируем контекст симуляции для управления физикой и временем
    sim_context = SimulationContext(stage_units_in_meters=True)
    sim_context.initialize_physics()

    print("Склад успешно загружен! Симуляция запущена.")

    # 6. Основной цикл симуляции (держит окно открытым и обновляет кадры)
    while simulation_app.is_running():
        # Шаг симуляции (физика + рендеринг)
        sim_context.step(render=True)

    # 7. Корректное закрытие приложения при выходе
    simulation_app.close()

if __name__ == "__main__":
    main()
