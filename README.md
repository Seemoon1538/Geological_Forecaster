
# Geological Forecaster

## Описание проекта (Русский)

**Geological Forecaster** — это Python-пайплайн для анализа геологических данных месторождений, таких как Homestake Mine в Южной Дакоте. Проект обрабатывает данные USGS в форматах Shapefile, GeoJSON или WMS, выполняет кластеризацию с помощью DBSCAN, прогнозирует объём золота и создаёт визуализации (интерактивные карты, тепловые карты, графики кластеров).

### Основные возможности
- Загрузка и обработка геологических данных (Shapefile, GeoJSON, WMS).
- Фильтрация точек в районе Homestake Mine (~[44.36, -103.75]).
- Кластеризация с помощью DBSCAN (`eps=0.005`, `min-samples=2`).
- Генерация синтетических данных (объём золота 7–10 г/т, тип руды "vein"), если реальные данные отсутствуют.
- Вывод результатов:
  - `forecast.json`: координаты центра месторождения, кластеры, прогнозируемый объём золота.
  - `interactive_map.html`: интерактивная карта с кластерами (Folium).
  - `clusters.png` и `heatmap.png`: визуализации кластеров и плотности.

### Пример результата
- Центр месторождения: `[44.3477, -103.7548]` (Homestake Mine).
- Прогнозируемый объём: 94.54 г/т Au (на основе синтетических данных).
- Тип руды: жильная ("vein").

## Установка (Русский)

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/Seemoon1538/Geological_Forecaster.git
   cd Geological_Forecaster

   
2. Создайте виртуальное окружение и активируйте его:
   ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    .venv\Scripts\activate     # Windows

3.Установите зависимости:
    ```bash
    
      pip install -r requirements.txt

(Опционально) Скачайте данные USGS:Загрузите Shapefile для Южной Дакоты с mrdata.usgs.gov/usmin/download.php.
Поместите файлы (например, SD-mine.shp) в папку data/.

Использование 
1.Запустите анализ с Shapefile:
     ```bash

    python -m src.main --input data/SD-point.shp --output output_homestake --eps 0.005 --min-samples 2

2.Используйте WMS для фона карты:
     ```bash

    python -m src.main --input https://basemap.nationalmap.gov/arcgis/services/USGSTopo/MapServer/WMSServer --output output_homestake_wms --wms-layer USGS_Topo --bbox -103.8 44.3 -103.7 44.4 --eps 0.005 --min-samples 2

3.Проверьте результаты:

output_homestake/forecast.json: координаты и объём золота.
output_homestake/interactive_map.html: интерактивная карта.
output_homestake/clusters.png и heatmap.png: визуализации.



Примечания 
Текущий анализ использует синтетические данные (объём 7–10 г/т Au, тип "vein") для демонстрации.
Для реальных данных используйте USGS Shapefile (SD-mine.shp) с полями GRADE_AU, DEPOSIT_TYPE.
Результаты оптимизированы для Homestake Mine, но пайплайн применим к любым геологическим данным.

Лицензия 
MIT License


Project Description 
Geological Forecaster is a Python pipeline for analyzing geological data from deposits like Homestake Mine in South Dakota. It processes USGS data in Shapefile, GeoJSON, or WMS formats, performs DBSCAN clustering, predicts gold volume, and generates visualizations (interactive maps, heatmaps, cluster plots).Key FeaturesLoads and processes geological data (Shapefile, GeoJSON, WMS).
Filters points to the Homestake Mine region (~[44.36, -103.75]).
Applies DBSCAN clustering (eps=0.005, min-samples=2).
Generates synthetic data (gold volume 7–10 g/t, ore type "vein") if real data is missing.
Outputs:forecast.json: deposit center, clusters, and predicted gold volume.
interactive_map.html: interactive map with Folium.
clusters.png and heatmap.png: cluster and density visualizations.

Example OutputDeposit center: [44.3477, -103.7548] (Homestake Mine).
Predicted volume: 94.54 g/t Au (based on synthetic data).
Ore type: vein.

Installation 
1.Clone the repository:
    ```bash
    
    git clone https://github.com/Seemoon1538/Geological_Forecaster.git
    cd Geological_Forecaster

2.Create and activate a virtual environment:
    ```bash
    
    python -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    .venv\Scripts\activate     # Windows

3.Install dependencies:
    ```bash

    pip install -r requirements.txt

(Optional) Download USGS data:Download a South Dakota Shapefile from mrdata.usgs.gov/usmin/download.php.
Place files (e.g., SD-mine.shp) in the data/ directory.

Usage 
1.Run analysis with a Shapefile:
    ```bash
    
    python -m src.main --input data/SD-point.shp --output output_homestake --eps 0.005 --min-samples 2

2.Use WMS for map background:
    ```bash

    python -m src.main --input https://basemap.nationalmap.gov/arcgis/services/USGSTopo/MapServer/WMSServer --output output_homestake_wms --wms-layer USGS_Topo --bbox -103.8 44.3 -103.7 44.4 --eps 0.005 --min-samples 2

3.Check results:
output_homestake/forecast.json: coordinates and gold volume.
output_homestake/interactive_map.html: interactive map.
output_homestake/clusters.png and heatmap.png: visualizations.



Notes 
Current analysis uses synthetic data (volume 7–10 g/t Au, ore type "vein") for demonstration.
For real data, use a USGS Shapefile (SD-mine.shp) with GRADE_AU, DEPOSIT_TYPE fields.
Results are optimized for Homestake Mine but applicable to any geological dataset.

License 
MIT License



