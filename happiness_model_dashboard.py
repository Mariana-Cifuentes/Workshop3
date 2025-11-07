import os
import warnings
import numpy as np
import pandas as pd
import mysql.connector as mysql
from dotenv import load_dotenv

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State

from scipy.stats import norm

load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning)

# DB Config

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASS = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DB   = os.getenv("MYSQL_DB", "etl_workshop3")

# Utilities

def get_connection():
    return mysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASS,
        database=MYSQL_DB,
        connection_timeout=5,
        raise_on_warnings=True
    )

def load_predictions():
    q = """
    SELECT
      c.country_name   AS country,
      t.year           AS year,
      f.score_actual   AS y_true,
      f.score_pred     AS y_pred,
      f.is_train, f.is_test,
      f.gdp_per_capita,
      f.healthy_life_expectancy,
      f.social_support,
      f.freedom,
      f.generosity,
      f.perceptions_of_corruption
    FROM fact_predictions f
    JOIN dim_country c ON c.country_id = f.country_id
    JOIN dim_time    t ON t.time_id    = f.time_id
    WHERE f.score_pred IS NOT NULL AND f.score_actual IS NOT NULL
    """
    try:
        cn = get_connection()
        df = pd.read_sql(q, cn)
        cn.close()
    except Exception as e:
        print(f"Could not load from MySQL ({type(e).__name__}): {e}")
        return pd.DataFrame(columns=[
            "country","year","y_true","y_pred","is_train","is_test",
            "gdp_per_capita","healthy_life_expectancy","social_support",
            "freedom","generosity","perceptions_of_corruption",
            "residual","abs_error","split"
        ])

    df = df.dropna(subset=["y_true", "y_pred"]).copy()

    def tag_split(row):
        if row.get("is_test", 0) == 1:  return "Test"
        if row.get("is_train", 0) == 1: return "Train"
        return "Unknown"

    df["split"] = df.apply(tag_split, axis=1)
    df["residual"]  = df["y_true"] - df["y_pred"]
    df["abs_error"] = df["residual"].abs()
    return df

def r2_score(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)

def mae(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def compute_kpis(df):
    if df.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    R2   = r2_score(df["y_true"], df["y_pred"])
    MAE_ = mae(df["y_true"], df["y_pred"])
    RMSE_ = rmse(df["y_true"], df["y_pred"])
    MAPE_ = mape(df["y_true"], df["y_pred"])
    BIAS = float((df["y_pred"] - df["y_true"]).mean())
    return R2, MAE_, RMSE_, MAPE_, BIAS

# Country name normalization for Plotly names-mode

_ALIASES = {
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "congo (brazzaville)": "Republic of the Congo",
    "congo (kinshasa)": "Democratic Republic of the Congo",
    "czech republic": "Czechia",
    "macedonia": "North Macedonia",
    "trinidad & tobago": "Trinidad and Tobago",
    "palestinian territories": "Palestine",
    "palestine": "Palestine",
    "south korea": "South Korea",
    "north korea": "North Korea",
    "laos": "Laos",
    "russia": "Russia",
    "taiwan": "Taiwan",
    "hong kong s.a.r. of china": "Hong Kong",
}

def normalize_country_for_plotly(name: str) -> str:
    if pd.isna(name):
        return ""
    n = str(name).strip()
    if not n:
        return ""
    low = n.lower()
    if low in _ALIASES:
        return _ALIASES[low]
    if "&" in n:
        n = n.replace("&", "and")
    return " ".join([w.capitalize() if w.isalpha() else w for w in n.split()])

# Assumption diagnostics

def durbin_watson_residuals(residuals_sorted: np.ndarray) -> float:
    if residuals_sorted.size < 2:
        return np.nan
    diff = np.diff(residuals_sorted)
    num = np.sum(diff**2)
    den = np.sum(residuals_sorted**2)
    return float(num/den) if den != 0 else np.nan

def compute_dw(df: pd.DataFrame) -> float:
    dfo = df.sort_values(["year", "country"], kind="mergesort")
    e = dfo["residual"].to_numpy(dtype=float)
    try:
        from statsmodels.stats.stattools import durbin_watson as sm_dw
        return float(sm_dw(e))
    except Exception:
        return durbin_watson_residuals(e)

def make_qq_data(residuals: np.ndarray):
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return np.array([]), np.array([])
    r_sorted = np.sort(r)
    probs = (np.arange(1, n+1) - 0.5) / n
    theo = norm.ppf(probs)
    return theo, r_sorted


# VIF

def compute_vif_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["gdp_per_capita", "social_support", "healthy_life_expectancy",
            "freedom", "generosity", "perceptions_of_corruption"]
    cols = [c for c in cols if c in df.columns]
    if len(cols) == 0 or df.empty:
        return pd.DataFrame(columns=["Variable","VIF"])

    X = df[cols].to_numpy(dtype=float)
    mask_valid = np.isfinite(X).all(axis=0) & (X.std(axis=0) > 0)
    cols = [c for c, m in zip(cols, mask_valid) if m]
    X = X[:, mask_valid]
    n, k = X.shape
    if k == 0:
        return pd.DataFrame(columns=["Variable","VIF"])

    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X_sm = sm.add_constant(X)
        vifs = [variance_inflation_factor(X_sm, i+1) for i in range(k)]
        return pd.DataFrame({"Variable": cols, "VIF": vifs}).sort_values("VIF", ascending=False)
    except Exception:
        vifs = []
        for j in range(k):
            y = X[:, j]
            X_others = np.delete(X, j, axis=1)
            Xo = np.column_stack([np.ones(n), X_others])
            beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
            yhat = Xo @ beta
            ss_res = np.sum((y - yhat)**2)
            ss_tot = np.sum((y - y.mean())**2)
            R2 = 1.0 - ss_res/ss_tot if ss_tot != 0 else 0.0
            vif = np.inf if (1.0 - R2) <= 0 else 1.0/(1.0 - R2)
            vifs.append(vif)
        return pd.DataFrame({"Variable": cols, "VIF": vifs}).sort_values("VIF", ascending=False)


# App (defer DB load at startup)

df_all = pd.DataFrame(columns=[
    "country","year","y_true","y_pred","is_train","is_test",
    "gdp_per_capita","healthy_life_expectancy","social_support",
    "freedom","generosity","perceptions_of_corruption",
    "residual","abs_error","split"
])

app = Dash(__name__, title="Happiness Model Performance")

def _kpi_style():
    return {"border":"1px solid #eee", "borderRadius":"10px", "padding":"12px 16px",
            "minWidth":"160px", "background":"#fff", "boxShadow":"0 1px 4px rgba(0,0,0,0.06)"}
def _kpi_style_small():
    s = _kpi_style(); s["minWidth"] = "120px"; return s
def _kpi_title():
    return {"fontSize":"12px", "color":"#777", "marginBottom":"6px",
            "textTransform":"uppercase", "letterSpacing":"0.04em"}
def _kpi_value():
    return {"fontFamily":"monospace", "fontSize":"20px", "fontWeight":"700"}

app.layout = html.Div(className="container", children=[
    html.H2("Happiness Score — Model Performance Dashboard", style={"marginTop":"10px"}),

    html.Div(style={"display":"flex", "gap":"12px", "flexWrap":"wrap"}, children=[
        html.Div(children=[
            html.Label("Dataset Split"),
            dcc.Dropdown(
                id="split-dd",
                options=[{"label":"Test (recommended)", "value":"Test"},
                         {"label":"Train", "value":"Train"},
                         {"label":"All", "value":"All"}],
                value="Test", clearable=False, style={"minWidth":"200px"}
            )
        ]),
        html.Div(children=[
            html.Label("Year"),
            dcc.Dropdown(
                id="year-dd",
                options=[{"label":"All", "value":"All"}] + [{"label":str(y), "value":y} for y in range(2015, 2020)],
                value="All", clearable=False, style={"minWidth":"160px"}
            )
        ]),
        html.Button("Reload Data", id="reload-btn", n_clicks=0, style={"height":"38px", "alignSelf":"end"}),
    ]),

    html.Hr(),

    # KPIs
    html.Div(id="kpi-cards", style={"display":"flex", "gap":"20px", "flexWrap":"wrap"}),

    # Scatter and Residuals distribution
    html.Div(style={"display":"grid", "gridTemplateColumns":"1fr 1fr", "gap":"16px", "marginTop":"10px"}, children=[
        dcc.Graph(id="scatter-actual-vs-pred"),
        dcc.Graph(id="hist-residuals"),
    ]),

    # Observed vs Predicted (series comparison)
    html.Div(style={"display":"grid", "gridTemplateColumns":"1fr", "gap":"16px"}, children=[
        dcc.Graph(id="series-obs-vs-pred"),
    ]),

    # Error by year and R² by year
    html.Div(style={"display":"grid", "gridTemplateColumns":"1fr 1fr", "gap":"16px"}, children=[
        dcc.Graph(id="bar-error-by-year"),
        dcc.Graph(id="r2-by-year"),
    ]),

    # Error boxplot and Overfitting check
    html.Div(style={"display":"grid", "gridTemplateColumns":"1fr 1fr", "gap":"16px"}, children=[
        dcc.Graph(id="box-abs-error"),
        dcc.Graph(id="bar-error-by-split"),
    ]),

    # Assumptions — Residuals vs Pred and QQ-plot
    html.Div(style={"display":"grid", "gridTemplateColumns":"1fr 1fr", "gap":"16px"}, children=[
        dcc.Graph(id="residuals-vs-pred"),
        dcc.Graph(id="qq-plot-residuals"),
    ]),

    # MAE by country map + Top-10 absolute errors + VIF
    html.Div(style={"display":"grid", "gridTemplateColumns":"1.2fr 0.8fr", "gap":"16px"}, children=[
        dcc.Graph(id="map-mae-country"),
        html.Div(children=[
            html.H4("Top-10 Absolute Errors"),
            dash_table.DataTable(
                id="table-top-errors",
                columns=[
                    {"name":"Country", "id":"country"},
                    {"name":"Year", "id":"year"},
                    {"name":"Actual (y)", "id":"y_true", "type":"numeric", "format": {"specifier":".4f"}},
                    {"name":"Predicted (ŷ)", "id":"y_pred", "type":"numeric", "format": {"specifier":".4f"}},
                    {"name":"|Error|", "id":"abs_error", "type":"numeric", "format": {"specifier":".4f"}},
                    {"name":"Residual", "id":"residual", "type":"numeric", "format": {"specifier":".4f"}},
                ],
                page_size=10,
                sort_action="native",
                style_table={"overflowX":"auto"},
                style_cell={"fontFamily":"monospace", "fontSize":"13px", "padding":"6px"},
                style_header={"backgroundColor":"#f6f6f6", "fontWeight":"bold"}
            ),
            html.H4("VIF (Multicollinearity)"),
            dash_table.DataTable(
                id="table-vif",
                columns=[{"name":"Variable","id":"Variable"}, {"name":"VIF","id":"VIF","type":"numeric","format":{"specifier":".3f"}}],
                page_size=6,
                sort_action="native",
                style_table={"overflowX":"auto","marginTop":"8px"},
                style_cell={"fontFamily":"monospace","fontSize":"13px","padding":"6px"},
                style_header={"backgroundColor":"#f6f6f6","fontWeight":"bold"}
            )
        ])
    ]),

    html.Div(style={"marginTop":"8px", "fontSize":"12px", "color":"#666"},
             children="Note: By default, metrics are shown for the TEST split (tracked by is_test=1).")
])

# Helpers

def filter_df(base, split_sel, year_sel):
    df = base.copy()
    if split_sel == "Test":
        df = df[df["split"] == "Test"]
    elif split_sel == "Train":
        df = df[df["split"] == "Train"]
    if year_sel != "All":
        df = df[df["year"] == int(year_sel)]
    return df

def safe_bias(b):
    return 0.0 if (pd.isna(b) or abs(b) < 1e-6) else float(b)


# Callbacks

@app.callback(
    Output("kpi-cards", "children"),
    Output("scatter-actual-vs-pred", "figure"),
    Output("hist-residuals", "figure"),
    Output("series-obs-vs-pred", "figure"),
    Output("bar-error-by-year", "figure"),
    Output("r2-by-year", "figure"),
    Output("box-abs-error", "figure"),
    Output("bar-error-by-split", "figure"),
    Output("residuals-vs-pred", "figure"),
    Output("qq-plot-residuals", "figure"),
    Output("map-mae-country", "figure"),
    Output("table-top-errors", "data"),
    Output("table-vif", "data"),
    Input("split-dd", "value"),
    Input("year-dd", "value"),
    Input("reload-btn", "n_clicks")
)
def update_view(split_sel, year_sel, n_clicks):
    base = load_predictions() if n_clicks else df_all
    df = filter_df(base, split_sel, year_sel)

    R2, MAE_val, RMSE_val, MAPE_val, BIAS_raw = compute_kpis(df)
    BIAS = safe_bias(BIAS_raw)

    DW = compute_dw(df) if not df.empty else np.nan

    kpi_cards = [
        html.Div(style=_kpi_style(), children=[
            html.Div("R²", style=_kpi_title()),
            html.Div(f"{R2:0.6f}" if pd.notna(R2) else "N/A", style=_kpi_value())
        ]),
        html.Div(style=_kpi_style(), children=[
            html.Div("MAE", style=_kpi_title()),
            html.Div(f"{MAE_val:0.6f}" if pd.notna(MAE_val) else "N/A", style=_kpi_value())
        ]),
        html.Div(style=_kpi_style(), children=[
            html.Div("RMSE", style=_kpi_title()),
            html.Div(f"{RMSE_val:0.6f}" if pd.notna(RMSE_val) else "N/A", style=_kpi_value())
        ]),
        html.Div(style=_kpi_style(), children=[
            html.Div("MAPE (%)", style=_kpi_title()),
            html.Div(f"{MAPE_val:0.2f}%" if pd.notna(MAPE_val) else "N/A", style=_kpi_value())
        ]),
        html.Div(style=_kpi_style_small(), children=[
            html.Div("Bias (ŷ−y)", style=_kpi_title()),
            html.Div(f"{BIAS:+0.3f}" if pd.notna(BIAS) else "N/A", style=_kpi_value())
        ]),
        html.Div(style=_kpi_style_small(), children=[
            html.Div("Durbin–Watson", style=_kpi_title()),
            html.Div(f"{DW:0.3f}" if pd.notna(DW) else "N/A", style=_kpi_value())
        ]),
        html.Div(style=_kpi_style_small(), children=[
            html.Div("# Records", style=_kpi_title()),
            html.Div(len(df), style=_kpi_value())
        ]),
    ]

    # Actual vs Predicted (with 45-degree line)
    fig_sc = go.Figure() if df.empty else px.scatter(
        df, x="y_true", y="y_pred", color="year",
        hover_data=["country","split"],
        title="Actual vs Predicted"
    )
    if not df.empty:
        min_v = float(min(df["y_true"].min(), df["y_pred"].min()))
        max_v = float(max(df["y_true"].max(), df["y_pred"].max()))
        fig_sc.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                    mode="lines", name="45°", line=dict(dash="dash")))

    # Residuals distribution
    fig_hist = go.Figure() if df.empty else px.histogram(df, x="residual", nbins=30, title="Residuals Distribution")
    if not df.empty:
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")

    # Observed vs Predicted (series comparison)
    if df.empty:
        fig_series = go.Figure()
    else:
        if year_sel != "All":
            comp = (df.groupby("country", as_index=False)
                      .agg(y_true=("y_true","mean"), y_pred=("y_pred","mean"))
                      .sort_values("y_true"))
            x_vals = comp["country"]
            x_title = "Country"
        else:
            comp = (df.groupby("year", as_index=False)
                      .agg(y_true=("y_true","mean"), y_pred=("y_pred","mean"))
                      .sort_values("year"))
            x_vals = comp["year"].astype(str)
            x_title = "Year"

        fig_series = go.Figure()
        fig_series.add_trace(go.Scatter(
            x=x_vals, y=comp["y_true"],
            mode="lines+markers", name="Observed (y)",
            marker=dict(size=6),
            line=dict(width=2, color="royalblue"),
            legendgroup="series",
        ))
        fig_series.add_trace(go.Scatter(
            x=x_vals, y=comp["y_pred"],
            mode="lines+markers", name="Predicted (ŷ)",
            marker=dict(size=6),
            line=dict(width=2, color="firebrick"),
            legendgroup="series",
        ))
        fig_series.update_layout(
            title="Observed (y) vs Predicted (ŷ)",
            xaxis_title=x_title,
            yaxis_title="Happiness Score",
            hovermode="x unified"
        )

    # Mean absolute error by year
    if df.empty:
        fig_bar_year = go.Figure()
    else:
        by_year_err = df.groupby("year", as_index=False)["abs_error"].mean()
        fig_bar_year = px.bar(by_year_err, x="year", y="abs_error",
                              title="Average Absolute Error by Year",
                              labels={"abs_error":"MAE (by year)","year":"Year"})

    # R² by year
    if df.empty:
        fig_r2 = go.Figure()
    else:
        try:
            yearly_r2 = (df.groupby("year")
                           .apply(lambda g: r2_score(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
                                  include_groups=False)
                           .reset_index(name="r2")
                           .sort_values("year"))
        except TypeError:
            yearly_r2 = (df.groupby("year")
                           .apply(lambda g: r2_score(g["y_true"].to_numpy(), g["y_pred"].to_numpy()))
                           .reset_index(name="r2")
                           .sort_values("year"))
        fig_r2 = px.bar(yearly_r2, x="year", y="r2",
                        title="R² by Year",
                        labels={"r2":"Coefficient of determination","year":"Year"},
                        range_y=[0,1])

    # Error distribution by split
    fig_box = go.Figure() if df.empty else px.box(
        df, x="split", y="abs_error", color="year",
        title="Absolute Error Distribution by Year",
        labels={"abs_error":"Absolute Error","split":"Split"}
    )

    # Overfitting check: MAE by split
    if base.empty:
        fig_split = go.Figure()
    else:
        df_year = base.copy()
        if year_sel != "All":
            df_year = df_year[df_year["year"] == int(year_sel)]
        by_split = df_year.groupby("split", as_index=False)["abs_error"].mean()
        fig_split = px.bar(by_split, x="split", y="abs_error",
                           color="split",
                           title="Mean Absolute Error by Split (Overfitting Check)",
                           labels={"abs_error":"MAE","split":"Split"})

    # Residuals vs Predictions
    fig_resid_pred = go.Figure() if df.empty else px.scatter(
        df, x="y_pred", y="residual", color="year",
        title="Residuals vs Predictions (linearity/homoscedasticity check)",
        labels={"y_pred":"Predicted (ŷ)", "residual":"Residual"}
    )
    if not df.empty:
        fig_resid_pred.add_hline(y=0, line_dash="dash", line_color="red")

    # Residuals QQ-plot
    if df.empty:
        fig_qq = go.Figure()
    else:
        theo, emp = make_qq_data(df["residual"].to_numpy(dtype=float))
        fig_qq = go.Figure()
        if theo.size > 0:
            fig_qq.add_trace(go.Scatter(x=theo, y=emp, mode="markers", name="Residuals"))
            qx1, qx3 = np.percentile(theo, [25,75])
            qy1, qy3 = np.percentile(emp, [25,75])
            slope = (qy3 - qy1) / (qx3 - qx1) if (qx3 - qx1) != 0 else 1.0
            intercept = qy1 - slope*qx1
            line_x = np.array([theo.min(), theo.max()])
            line_y = intercept + slope*line_x
            fig_qq.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="Reference", line=dict(dash="dash")))
        fig_qq.update_layout(title="Residuals QQ-plot (normality)",
                             xaxis_title="Theoretical Quantiles N(0,1)",
                             yaxis_title="Residual Quantiles")

    # MAE by country map (names-mode)
    if df.empty:
        fig_map = go.Figure()
    else:
        by_country = df.groupby("country", as_index=False)["abs_error"].mean()
        by_country["country_plotly"] = by_country["country"].apply(normalize_country_for_plotly)
        fig_map = px.scatter_geo(
            by_country,
            locations="country_plotly",
            locationmode="country names",
            size="abs_error",
            hover_name="country",
            title="Mean Absolute Error by Country (MAE)"
        )

    # Top-10 absolute errors table
    top = df.sort_values("abs_error", ascending=False).head(10)
    table_top = top[["country","year","y_true","y_pred","abs_error","residual"]].to_dict("records")

    # VIF table
    vif_df = compute_vif_table(df)
    table_vif = vif_df.to_dict("records")

    return (
        kpi_cards,
        fig_sc,
        fig_hist,
        fig_series,
        fig_bar_year,
        fig_r2,
        fig_box,
        fig_split,
        fig_resid_pred,
        fig_qq,
        fig_map,
        table_top,
        table_vif
    )

# Main

if __name__ == "__main__":
    try:
        tmp = load_predictions()
        if not tmp.empty:
            df_all = tmp
            print(f"Initial data loaded: {len(df_all)} rows.")
        else:
            print("Starting without data (use 'Reload Data').")
    except Exception as e:
        print(f"Starting without data due to initial load error: {e}")

    app.run(host="127.0.0.1", port=8051, debug=True)