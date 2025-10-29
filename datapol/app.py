import sys
import importlib
import pkgutil

# Fix for Python 3.14 compatibility
if not hasattr(pkgutil, 'find_loader'):
    pkgutil.find_loader = lambda name: importlib.util.find_spec(name) is not None

import logging
import time
from datetime import datetime

import dash
import duckdb
import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from folium.plugins import FastMarkerCluster
from shapely.geometry import shape

# ---------- настройки ----------
POLYGON_PATH = "/Users/mariakrivorotova/PycharmProjects/pybot/datapol/polygon_data.json"  # один geojson-файл

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# ---------- работа с одним полигоном ----------
def read_geometry_from_geojson(path: str):
    """Читает GeoJSON и возвращает (geojson_geometry_dict, shapely_geometry)."""
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        geom_dict = data["features"][0]["geometry"]
    elif isinstance(data, dict) and data.get("type") == "Feature":
        geom_dict = data["geometry"]
    else:
        geom_dict = data 

    geom = shape(geom_dict)
    if geom.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"Ожидался Polygon/MultiPolygon, а пришло: {geom.geom_type}")

    return geom_dict, geom


try:
    POLY_GEOJSON, POLY_GEOM = read_geometry_from_geojson(POLYGON_PATH)
    minx, miny, maxx, maxy = POLY_GEOM.bounds
    POLY_CENTER = ((miny + maxy) / 2.0, (minx + maxx) / 2.0)
    POLY_BOUNDS = [[miny, minx], [maxy, maxx]]
    logging.info(
        f"Полигон загружен: тип={POLY_GEOM.geom_type}, "
        f"bounds=({minx:.6f},{miny:.6f})–({maxx:.6f},{maxy:.6f})"
    )
except Exception as e:
    logging.error(f"Не удалось загрузить полигон: {e}")
    POLY_GEOJSON, POLY_GEOM, POLY_CENTER, POLY_BOUNDS = None, None, (52.260853, 104.282274), None

# ---------- загрузка данных ----------
def load_data():
    try:
        conn = duckdb.connect(database="data.duckdb", read_only=True)
        df = conn.execute("SELECT * FROM orders").df()
        count_orders = conn.sql("SELECT count(*) FROM orders").fetchone()[0]
        logging.info(f"Loaded {count_orders} orders from the database.")
        return conn, df
    except Exception as e:
        print(f"Error loading data: {e}")
        conn = duckdb.connect(database=":memory:", read_only=False)
        df = pd.DataFrame(columns=[
            "type_user", "category_name", "ship_date",
            "price_of_order", "type_of_payment",
            "latitude", "longitude"
        ])
        return conn, df

# ---------- dash app ----------
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap",
    ],
)
app.title = "Интерактивная Карта"

conn, df = load_data()

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id="type-user-dropdown",
                        options=[{"label": r[0], "value": r[0]} for r in
                                 conn.execute("SELECT DISTINCT type_user FROM orders").fetchall()]
                        if "type_user" in df.columns and len(df) > 0 else [],
                        multi=True,
                        placeholder="Тип пользователя",
                    ),
                ], className="filter-column"),

                html.Div([
                    dcc.Dropdown(
                        id="category-dropdown",
                        options=[{"label": r[0], "value": r[0]} for r in
                                 conn.execute("SELECT DISTINCT category_name FROM orders").fetchall()]
                        if "category_name" in df.columns and len(df) > 0 else [],
                        multi=True,
                        placeholder="Категория",
                    ),
                ], className="filter-column"),

                html.Div([
                    dcc.DatePickerRange(
                        id="date-range",
                        min_date_allowed=conn.execute("SELECT MIN(ship_date) FROM orders").fetchone()[0]
                        if "ship_date" in df.columns and len(df) > 0 else datetime(2020, 1, 1),
                        max_date_allowed=conn.execute("SELECT MAX(ship_date) FROM orders").fetchone()[0]
                        if "ship_date" in df.columns and len(df) > 0 else datetime(2025, 12, 31),
                        start_date=conn.execute("SELECT MIN(ship_date) FROM orders").fetchone()[0]
                        if "ship_date" in df.columns and len(df) > 0 else datetime(2020, 1, 1),
                        end_date=conn.execute("SELECT MAX(ship_date) FROM orders").fetchone()[0]
                        if "ship_date" in df.columns and len(df) > 0 else datetime(2025, 12, 31),
                        display_format="YYYY-MM-DD",
                        first_day_of_week=1,
                        start_date_placeholder_text="Начальная дата",
                        end_date_placeholder_text="Конечная дата",
                        className="date-range-picker",
                    ),
                ], className="filter-column date-filter"),

                html.Div([
                    dcc.Dropdown(
                        id="payment-dropdown",
                        options=[{"label": r[0], "value": r[0]} for r in
                                 conn.execute("SELECT DISTINCT type_of_payment FROM orders").fetchall()]
                        if "type_of_payment" in df.columns and len(df) > 0 else [],
                        multi=True,
                        placeholder="Способ оплаты",
                    ),
                ], className="filter-column"),

                html.Div([
                    dcc.Dropdown(
                        id="map-type-dropdown",
                        options=[
                            {"label": "Точки", "value": "points"},
                            {"label": "Тепловая карта", "value": "heatmap"},
                            {"label": "Кластеры", "value": "clusters"},
                        ],
                        value="clusters",
                        clearable=False,
                        placeholder="Тип отображения карты",
                    ),
                ], className="filter-column"),
            ], className="filter-row"),
        ], className="filter-card"),

        html.Div([
            html.Div(id="map-container", style={"height": "800px", "width": "100%"}),
        ], className="map-container"),

        html.Div(id="filtered-data-info", style={"display": "none"}),
    ], className="container"),
], style={"backgroundColor": "var(--background-color)"})


@app.callback(
    Output("map-container", "children"),
    [
        Input("type-user-dropdown", "value"),
        Input("category-dropdown", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("payment-dropdown", "value"),
        Input("map-type-dropdown", "value"),
    ],
)
def update_map(selected_users, selected_categories, start_date, end_date, selected_payments, map_type):
    start_time = time.time()

    sql_query = "SELECT * FROM orders WHERE 1=1"
    if selected_users:
        placeholders = ", ".join([f"'{v}'" for v in selected_users])
        sql_query += f" AND type_user IN ({placeholders})"
    if selected_categories:
        placeholders = ", ".join([f"'{v}'" for v in selected_categories])
        sql_query += f" AND category_name IN ({placeholders})"
    if start_date and end_date:
        sql_query += f" AND ship_date >= '{start_date}' AND ship_date <= '{end_date}'"
    if selected_payments:
        placeholders = ", ".join([f"'{v}'" for v in selected_payments])
        sql_query += f" AND type_of_payment IN ({placeholders})"

    try:
        filtered_df = conn.execute(sql_query).fetchdf()
    except Exception as e:
        print(f"SQL Error: {e}")
        filtered_df = pd.DataFrame(columns=[
            "type_user", "category_name", "ship_date",
            "price_of_order", "type_of_payment",
            "latitude", "longitude"
        ])

    logging.info(f"Query: {sql_query}")
    logging.info(f"Query returned {len(filtered_df)} records")

    # --- если данных нет, рисуем пустую карту, центр по полигону ---
    if len(filtered_df) == 0 or "latitude" not in filtered_df.columns or "longitude" not in filtered_df.columns:
        empty_fig = px.scatter_mapbox(lat=[POLY_CENTER[0]], lon=[POLY_CENTER[1]], zoom=12, height=800)
        empty_fig.update_layout(
            mapbox_style="carto-positron",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=800,
            dragmode="pan",
        )
        # контур полигона (если доступен)
        if POLY_GEOJSON:
            empty_fig.update_layout(
                mapbox_layers=[{
                    "sourcetype": "geojson",
                    "source": {"type": "Feature", "geometry": POLY_GEOJSON},
                    "type": "line",
                    "line": {"width": 2},
                }]
            )
        execution_time = time.time() - start_time
        logging.info(f"Построение пустой карты заняло {execution_time:.4f} секунд")
        return dcc.Graph(figure=empty_fig, style={"height": "800px"},
                         config={"scrollZoom": True, "doubleClick": "reset"})

    # ----------- clusters (folium) -----------
    if map_type == "clusters":
        center_lat, center_lon = POLY_CENTER

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

        if POLY_GEOJSON:
            gj = folium.GeoJson(
                {"type": "Feature", "geometry": POLY_GEOJSON},
                name="Полигон",
                style_function=lambda x: {"color": "#2a4b8d", "weight": 2, "fill": False},
            )
            gj.add_to(m)
            if POLY_BOUNDS:
                m.fit_bounds(POLY_BOUNDS)

        cluster_data = filtered_df[["latitude", "longitude", "type_user"]].values.tolist()
        callback = """
        function (row) {
            var lat = row[0]; var lng = row[1]; var type = row[2];
            var color = '#3f51b5';
            if (type === 'ЮЛ') color = '#ff7043';
            else if (type === 'ИП') color = '#2e7d32';
            return L.circleMarker(new L.LatLng(lat, lng), {
                radius: 4, color: color, fillColor: color, fillOpacity: 0.7
            });
        };
        """
        FastMarkerCluster(data=cluster_data, callback=callback).add_to(m)

        html_string = m._repr_html_()
        execution_time = time.time() - start_time
        logging.info(f"Построение карты кластеров заняло {execution_time:.4f} секунд")
        return html.Iframe(srcDoc=html_string, style={"width": "100%", "height": "800px", "border": "none"})

    # ----------- points / heatmap (plotly) -----------
    if "price_of_order" in filtered_df.columns:
        filtered_df["price_formatted"] = filtered_df["price_of_order"].apply(
            lambda x: f"₽{x:,}".replace(",", " ")
        )

    hover_data = {
        "type_user": True,
        "category_name": True,
        "ship_date": True,
        "price_formatted": True,
        "price_of_order": False,
        "type_of_payment": True,
        "latitude": False,
        "longitude": False,
    }

    if map_type == "points":
        fig = px.scatter_mapbox(
            filtered_df, lat="latitude", lon="longitude",
            hover_name="category_name", hover_data=hover_data,
            color="type_user",
            color_discrete_map={"ФЛ": "#5c6ac4", "ЮЛ": "#ff9800", "ИП": "#4caf50"},
            opacity=0.7, zoom=11, height=800,
            center={"lat": POLY_CENTER[0], "lon": POLY_CENTER[1]},
        )
        fig.update_traces(marker=dict(size=6))

    elif map_type == "heatmap":
        fig = px.density_mapbox(
            filtered_df, lat="latitude", lon="longitude",
            z="price_of_order" if "price_of_order" in filtered_df.columns else None,
            radius=10, zoom=11, height=800,
            color_continuous_scale=[
                [0, "blue"], [0.4, "blue"], [0.65, "lime"], [1.0, "red"],
            ],
            center={"lat": POLY_CENTER[0], "lon": POLY_CENTER[1]},
            opacity=0.8,
        )

    # общий стиль + контур полигона
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=800,
        legend=dict(title="Тип пользователя", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode="pan",
        mapbox_layers=[
            {
                "sourcetype": "geojson",
                "source": {"type": "Feature", "geometry": POLY_GEOJSON} if POLY_GEOJSON else {},
                "type": "line",
                "line": {"width": 2},
            }
        ] if POLY_GEOJSON else [],
    )

    execution_time = time.time() - start_time
    logging.info(f"Построение карты {map_type} заняло {execution_time:.4f} секунд")
    return dcc.Graph(
        figure=fig,
        style={"height": "800px"},
        config={"scrollZoom": True, "doubleClick": "reset"},
    )


if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple('localhost', 8050, app.server, use_reloader=True, use_debugger=True)
