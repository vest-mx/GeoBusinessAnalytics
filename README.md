# GeoBusinessAnalytics
Создание полноценной системы для анализа географического распределения бизнес-данных с возможностью интерактивной визуализации и фильтрации. Генерация тестовой выборки происходит в проекте. Это позволяет аналитикам использовать "песочницу" для отработки геопространственного анализа без доступа к реальным данным.

(https://python.org)
(https://dash.plotly.com)


## 🎯 О чем этот проект?

Система для генерации и визуализации тестовых бизнес-данных с географической привязкой. Позволяет анализировать распределение заказов по регионам, сегментировать клиентов и выявлять пространственные закономерности.
<img width="1500" height="715" alt="image" src="https://github.com/user-attachments/assets/c8de730e-d59c-4389-9593-c32816f55c42" />


### 🚀 Ключевые возможности

- **🗺️ Гибкие регионы** - рисуйте любые полигоны в Kepler.gl
- **🎲 Реалистичные данные** - умная генерация с хотспотами и бизнес-атрибутами  
- **📊 3 режима визуализации** - точки, тепловые карты, кластеры
- **🔍 Интерактивные фильтры** - сегментация по 5 параметрам

### Пример интерактивной карты распределения или заказов, где каждая точка - одна операция следующего вида:

<img width="1269" height="402" alt="image" src="https://github.com/user-attachments/assets/64e796f8-0513-4b46-ab31-906cb31bc7b6" />




## 🛠️ Быстрый старт

# 1. Установка зависимостей

git clone https://github.com/yourname/pybot-geo-analytics
cd pybot-geo-analytics

pip install -r requirements.txt

# 2. Подготовка данных

### Сгенерируйте тестовые данные
python datapol/gendata.py

### Или используйте готовый пример
python datapol/kepler_converter.py

# 3. Запуск приложения

python datapol/app.py

Running on http://localhost:8050   --- локальный запуск




# Генерация данных

### Создает распределение с "городами"
df = generate_points_in_polygon(
    polygon=polygon, 
    num_points=10_000, 
    with_hotspots=True
)

### Визуализация

- **🔴 Точечная карта** - цветовая кодировка по типу клиента
    
- **🔥 Тепловая карта** - интенсивность по стоимости заказов
    
- **👥 Кластеры** - группировка точек по плотности
    
### Фильтры

- Тип пользователя (ФЛ/ЮЛ/ИП)
    
- Категория товара
    
- Диапазон дат
    
- Способ оплаты
    
- Тип визуализации
    
## 🛠️ Технологический стек

|Компонент|Назначение|
|---|---|
|**Dash**|Веб-фреймворк для дашбордов|
|**Plotly**|Интерактивные графики|
|**Folium**|Визуализация карт|
|**DuckDB**|Аналитическая БД|
|**Shapely**|Геометрические операции|
|**Faker**|Генерация тестовых данных|





## 📄 Лицензия

MIT License - смотрите файл [LICENSE](https://license/)

## 👨‍💻 Автор

[GitHub](https://github.com/vest-mx)




---

## 📋 **ДОПОЛНИТЕЛЬНЫЕ ФАЙЛЫ**

### **1. requirements.txt (обновленный)**
```txt
# Основные
dash>=2.14.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0

# Геоданные
folium>=0.15.0
shapely>=2.0.0
geopandas>=0.13.0

# База данных
duckdb>=0.9.0

# Генерация данных
faker>=18.0.0
tqdm>=4.65.0

# Утилиты
werkzeug>=2.3.0
click>=8.1.0

2. .gitignore

gitignore
# Данные
*.duckdb
data.csv
temp_*

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

3. docs/setup_guide.md

# 📚 Полное руководство по установке

## 1. Клонирование репозитория
git clone https://github.com/yourname/pybot-geo-analytics
cd pybot-geo-analytics


2. Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


3. Установка зависимостей
pip install -r requirements.txt

4. Настройка полигона
Откройте Kepler.gl
Нарисуйте нужный регион
Сохраните как kepler.gl.json
Конвертируйте: python kepler_converter.py

5. Запуск

# Генерация данных
python datapol/gendata.py

# Запуск приложения  
python datapol/app.py
