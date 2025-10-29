import datetime
import json
import os

import duckdb
import numpy as np
import pandas as pd
from faker import Faker
from shapely.geometry import Point, shape
from tqdm import tqdm


# ---------- Геометрия ----------
def load_polygon_from_json(json_data):
    """Загрузка полигона из GeoJSON"""
    try:
        polygon_geom = shape(json_data)
        print(f"Загружен полигон типа: {polygon_geom.geom_type}")
        print(f"Площадь полигона: {polygon_geom.area:.6f} кв. градусов")
        print(f"Периметр полигона: {polygon_geom.length:.6f} градусов")
        return polygon_geom
    except Exception as e:
        print(f"Ошибка при загрузке полигона: {e}")
        return None


def calculate_center_of_polygon(polygon):
    """Вычисление центра полигона (lat, lon)"""
    centroid = polygon.centroid
    return centroid.y, centroid.x


# ---------- Генерация точек ----------
def generate_points_in_polygon(polygon=None, num_points=10_000, with_hotspots=True):
    """
    Генерация точек внутри ПОЛИГОНА (одного).
    Возвращает DataFrame с колонками ['latitude', 'longitude'].
    """
    print(f"Генерация {num_points} точек внутри полигона...")

    min_x, min_y, max_x, max_y = polygon.bounds
    center_y, center_x = calculate_center_of_polygon(polygon)

    points = []

    if with_hotspots:
        hotspots = [
            (center_y, center_x,                                 0.025, 0.30),  # центр
            (min_y + (max_y - min_y) * 0.75, center_x,           0.020, 0.20),  # север
            (min_y + (max_y - min_y) * 0.25, center_x,           0.020, 0.15),  # юг
            (center_y,                 min_x + (max_x - min_x) * 0.75, 0.015, 0.20),  # восток
            (center_y,                 min_x + (max_x - min_x) * 0.25, 0.015, 0.15),  # запад
        ]

        random_points_percent = 0.40
        hotspot_points = int(num_points * (1 - random_points_percent))
        random_points = num_points - hotspot_points

        hotspot_weights = [h[3] for h in hotspots]
        total_weight = sum(hotspot_weights)
        hotspot_targets = [int(hotspot_points * w / total_weight) for w in hotspot_weights]

        diff = hotspot_points - sum(hotspot_targets)
        if diff != 0:
            max_idx = hotspot_weights.index(max(hotspot_weights))
            hotspot_targets[max_idx] += diff

        for idx, (hot_y, hot_x, radius, _) in enumerate(hotspots):
            target = hotspot_targets[idx]
            with tqdm(total=target, desc=f"Хотспот {idx+1}") as pbar:
                generated = 0
                attempts = 0
                max_attempts = target * 50
                while generated < target and attempts < max_attempts:
                    dx = np.random.normal(0, radius)
                    dy = np.random.normal(0, radius)
                    x, y = hot_x + dx, hot_y + dy
                    if polygon.contains(Point(x, y)):
                        points.append((y, x))  # (lat, lon)
                        generated += 1
                        pbar.update(1)
                    attempts += 1

        print("\nГенерация точек с равномерным распределением...")
        with tqdm(total=random_points) as pbar:
            generated = 0
            attempts = 0
            max_attempts = random_points * 50
            while generated < random_points and attempts < max_attempts:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                if polygon.contains(Point(x, y)):
                    points.append((y, x))
                    generated += 1
                    pbar.update(1)
                attempts += 1
    else:
        print("Генерация точек с равномерным распределением...")
        with tqdm(total=num_points) as pbar:
            generated = 0
            attempts = 0
            max_attempts = num_points * 50
            while generated < num_points and attempts < max_attempts:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                if polygon.contains(Point(x, y)):
                    points.append((y, x))
                    generated += 1
                    pbar.update(1)
                attempts += 1

    print(f"Сгенерировано {len(points)} точек")
    return pd.DataFrame(points, columns=["latitude", "longitude"])


# ---------- Обогащение ----------
def enrich_dataframe(df):
    """
    Обогащает DataFrame случайными бизнес-полями
    """
    df_enriched = df.copy()
    fake = Faker()

    value_dict = {
        "type_user": ("ЮЛ", "ИП", "ФЛ"),
        "category_name": (
            "Напитки", "Приправы и соусы", "Кондитерские изделия", "Молочные продукты",
            "Крупы и злаки", "Мясо и птица", "Овощи и фрукты", "Морепродукты"
        ),
        "type_of_payment": ("Наличные", "Карта", "QR-код", "Кредит", "Счёт"),
    }

    n = len(df_enriched)
    df_enriched["type_user"] = np.random.choice(value_dict["type_user"], size=n)
    df_enriched["category_name"] = np.random.choice(value_dict["category_name"], size=n)

    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2025, 1, 1)
    df_enriched["ship_date"] = [
        fake.date_time_ad(start_datetime=start_date, end_datetime=end_date).strftime("%Y-%m-%d")
        for _ in range(n)
    ]

    df_enriched["price_of_order"] = np.random.randint(100, 100_001, size=n, dtype=np.int64)

    df_enriched["type_of_payment"] = np.random.choice(value_dict["type_of_payment"], size=n)

    return df_enriched


# ---------- Основной сценарий ----------
if __name__ == "__main__":
    file_path ='/Users/mariakrivorotova/PycharmProjects/pybot/datapol/polygon_data.json'
    with open(file_path, encoding="utf-8") as f:
        polygon_data = json.load(f)
    polygon = load_polygon_from_json(polygon_data)

    df = generate_points_in_polygon(polygon=polygon, num_points=10_000, with_hotspots=True)
    df = enrich_dataframe(df)

    csv_file = "data.csv"
    df.to_csv(csv_file, index=False)

    conn = duckdb.connect("../data.duckdb")
    conn.sql(
        """
        CREATE TABLE IF NOT EXISTS orders (
            latitude DOUBLE,
            longitude DOUBLE,
            type_user VARCHAR,
            category_name VARCHAR,
            ship_date DATE,
            price_of_order BIGINT,
            type_of_payment VARCHAR
        );
        """
    )
    conn.sql(
        """
        INSERT INTO orders
        SELECT * FROM read_csv_auto('data.csv');
        """
    )
    conn.close()

    if os.path.exists(csv_file):
        os.remove(csv_file)
        print(f"Файл '{csv_file}' успешно удален.")
    else:
        print(f"Файл '{csv_file}' не существует.")
