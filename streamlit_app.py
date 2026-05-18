import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr

st.set_page_config(
    page_title="Dashboard Bienestar Laboral",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# Configuración general
# ============================================================

POSSIBLE_DATA_PATHS = [
    Path("data/bienestar_laboral_EDA.xlsx"),
    Path("bienestar_laboral_EDA.xlsx"),
    Path(__file__).parent / "data" / "bienestar_laboral_EDA.xlsx",
    Path(__file__).parent / "bienestar_laboral_EDA.xlsx",
]

likert_1_5 = {
    "Nunca": 1,
    "Rara vez": 2,
    "Raramente": 2,
    "Algunas veces": 3,
    "Frecuentemente": 4,
    "A menudo": 4,
    "Casi siempre": 4,
    "Siempre": 5,
}

agreement_1_7 = {
    "Muy en desacuerdo": 1,
    "Moderadamente en desacuerdo": 2,
    "Algo en desacuerdo": 3,
    "Ni de acuerdo ni en desacuerdo": 4,
    "Neutral": 4,
    "Algo de acuerdo": 5,
    "Moderadamente de acuerdo": 6,
    "Muy de acuerdo": 7,
}

freq_1_7 = {
    "Nunca": 1,
    "Raramente": 2,
    "Rara vez": 2,
    "Ocasionalmente": 3,
    "Algunas veces": 4,
    "Frecuentemente": 5,
    "A menudo": 5,
    "Casi siempre": 6,
    "Siempre": 7,
}

dimensions = {
    "CTRL": {"label": "Control del trabajo", "items": ["CT1", "CT2", "CT3"], "scale": "1-5", "type": "resource"},
    "PT": {"label": "Presión del tiempo", "items": ["PT1", "PT2", "PT3", "PT4"], "scale": "1-5", "type": "risk"},
    "CL": {"label": "Compromiso del líder", "items": [f"CL{i}" for i in range(1, 8)], "scale": "1-5", "type": "resource"},
    "AC": {"label": "Apoyo de compañeros", "items": ["AC1", "AC2", "AC3"], "scale": "1-5", "type": "resource"},
    "CR": {"label": "Claridad de rol", "items": [f"CR{i}" for i in range(1, 5)], "scale": "1-5", "type": "resource"},
    "CoR": {"label": "Conflicto de rol", "items": ["CoR1", "CoR2", "CoR3"], "scale": "1-5", "type": "risk"},
    "GC": {"label": "Gestión del cambio", "items": [f"GC{i}" for i in range(1, 5)], "scale": "1-5", "type": "resource"},
    "SM": {"label": "Salud mental organizacional", "items": [f"SM{i}" for i in range(1, 6)], "scale": "1-5", "type": "resource"},
    "SAT": {"label": "Satisfacción / Engagement", "items": [f"SAT{i}" for i in range(1, 10)], "scale": "1-7", "type": "resource"},
    "IR": {"label": "Intención de retiro", "items": [f"IR{i}" for i in range(1, 5)], "scale": "1-7", "type": "risk"},
    "FT": {"label": "Conflicto Familia → Trabajo", "items": [f"FT{i}" for i in range(1, 6)], "scale": "1-7", "type": "risk"},
    "TF": {"label": "Conflicto Trabajo → Familia", "items": [f"TF{i}" for i in range(1, 6)], "scale": "1-7", "type": "risk"},
    "BU": {"label": "Burnout / Agotamiento", "items": [f"BU{i}" for i in range(1, 13)], "scale": "1-5", "type": "risk"},
    "BP": {"label": "Bienestar percibido", "items": [f"BP{i}" for i in range(1, 11)], "scale": "1-7", "type": "resource"},
    "SOM": {"label": "Somatización", "items": [f"SOM{i}" for i in range(1, 6)], "scale": "1-7-freq", "type": "risk"},
    "DL": {"label": "Desgaste laboral", "items": [f"DL{i}" for i in range(1, 9)], "scale": "1-7-freq", "type": "risk"},
}

scale_ranges = {
    "1-5": (1, 5),
    "1-7": (1, 7),
    "1-7-freq": (1, 7),
}

dimension_cols = list(dimensions.keys())
filter_cols = ["Sexo", "Tipo_Cargo", "Sector", "Modalidad"]


# ============================================================
# Funciones auxiliares
# ============================================================

def find_data_path() -> Path:
    for path in POSSIBLE_DATA_PATHS:
        if path.exists():
            return path
    st.error(
        "No se encontró el archivo `bienestar_laboral_EDA.xlsx`. "
        "Déjalo en la misma carpeta de esta app o dentro de una carpeta llamada `data`."
    )
    st.stop()


def scale_to_100(series: pd.Series, scale_name: str) -> pd.Series:
    lo, hi = scale_ranges[scale_name]
    return ((series - lo) / (hi - lo)) * 100


def urgency_score(mean_value: float, meta: dict) -> float:
    lo, hi = scale_ranges[meta["scale"]]
    normalized = ((mean_value - lo) / (hi - lo)) * 100

    # En variables de riesgo, valores altos = más urgencia.
    # En variables de recurso, valores bajos = más urgencia.
    if meta["type"] == "risk":
        return normalized
    return 100 - normalized


def corr_value(df: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    data = df[[x, y]].dropna()
    if len(data) < 3:
        return float("nan"), float("nan")
    rho, p_value = spearmanr(data[x], data[y])
    return rho, p_value


def corr_strength(rho: float) -> str:
    abs_rho = abs(rho)
    if pd.isna(abs_rho):
        return "no calculable"
    if abs_rho < 0.20:
        return "muy débil"
    if abs_rho < 0.40:
        return "débil"
    if abs_rho < 0.60:
        return "moderada"
    if abs_rho < 0.80:
        return "fuerte"
    return "muy fuerte"


def corr_direction(rho: float) -> str:
    if pd.isna(rho):
        return "no calculable"
    return "positiva" if rho > 0 else "negativa" if rho < 0 else "nula"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Codificación Likert
    for dim, meta in dimensions.items():
        if meta["scale"] == "1-5":
            mapper = likert_1_5
        elif meta["scale"] == "1-7":
            mapper = agreement_1_7
        else:
            mapper = freq_1_7

        for item in meta["items"]:
            if item in out.columns:
                out[item] = pd.to_numeric(out[item].replace(mapper), errors="coerce")

    # Inversión psicométrica de IR1.
    if "IR1" in out.columns:
        out["IR1_inv"] = 8 - out["IR1"]

    # Construcción eficiente de dimensiones promedio.
    dimension_scores = {}
    for dim, meta in dimensions.items():
        items = [
            "IR1_inv" if dim == "IR" and item == "IR1" else item
            for item in meta["items"]
        ]
        existing_items = [item for item in items if item in out.columns]

        if existing_items:
            dimension_scores[dim] = out[existing_items].mean(axis=1)

    out = pd.concat([out, pd.DataFrame(dimension_scores, index=out.index)], axis=1)
    return out.copy()


@st.cache_data(show_spinner="Cargando y preparando datos...")
def load_data() -> pd.DataFrame:
    data_path = find_data_path()
    return prepare(pd.read_excel(data_path))


def build_risk_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dim, meta in dimensions.items():
        if dim not in data.columns:
            continue

        mean_value = data[dim].mean()
        median_value = data[dim].median()
        std_value = data[dim].std()
        urgency = urgency_score(mean_value, meta)

        rows.append({
            "Código": dim,
            "Dimensión": meta["label"],
            "Tipo": "Riesgo" if meta["type"] == "risk" else "Recurso",
            "Media": mean_value,
            "Mediana": median_value,
            "Desv. estándar": std_value,
            "Urgencia 0-100": urgency,
        })

    return pd.DataFrame(rows).sort_values("Urgencia 0-100", ascending=False)


def filtered_by_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")

    st.sidebar.caption(
        "Cada filtro muestra todos los valores existentes en el dataset original. "
        "Selecciona `Todos` para no restringir esa variable."
    )

    filtered = df.copy()

    for col in filter_cols:
        if col not in df.columns:
            continue

        # IMPORTANTE:
        # Las opciones salen SIEMPRE del dataset completo, no del dataframe ya filtrado.
        # Así no desaparecen categorías cuando se aplica un filtro previo.
        options = sorted([str(x) for x in df[col].dropna().unique()])
        selected = st.sidebar.multiselect(
            label=col,
            options=["Todos"] + options,
            default=["Todos"],
            key=f"filter_{col}",
        )

        if "Todos" not in selected and selected:
            filtered = filtered[filtered[col].astype(str).isin(selected)]

    st.sidebar.divider()
    st.sidebar.metric("Registros filtrados", len(filtered))
    st.sidebar.metric("Registros totales", len(df))

    return filtered


def write_dynamic_story(data: pd.DataFrame, risk_df: pd.DataFrame) -> None:
    if data.empty:
        st.warning("No hay registros para los filtros seleccionados.")
        return

    top_risks = risk_df.head(3)
    top_resources = (
        risk_df[risk_df["Tipo"] == "Recurso"]
        .sort_values("Urgencia 0-100", ascending=False)
        .head(3)
    )

    rho_sat_ir, p_sat_ir = corr_value(data, "SAT", "IR")
    rho_bu_dl, p_bu_dl = corr_value(data, "BU", "DL")
    rho_cl_sat, p_cl_sat = corr_value(data, "CL", "SAT")
    rho_cl_bu, p_cl_bu = corr_value(data, "CL", "BU")

    st.markdown("### Lectura ejecutiva de los datos")

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown(
            f"""
            En la muestra filtrada se analizan **{len(data)} trabajadores**. 
            La historia principal del dashboard es identificar **dónde está el riesgo psicosocial**, 
            qué recursos organizacionales pueden estar protegiendo a los trabajadores y cómo se relacionan 
            las variables críticas de bienestar laboral.

            La dimensión con mayor urgencia actual es **{top_risks.iloc[0]["Dimensión"]}**
            con un puntaje de urgencia de **{top_risks.iloc[0]["Urgencia 0-100"]:.1f}/100**.
            """
        )

    with c2:
        st.markdown("**Tres prioridades de intervención**")
        for _, row in top_risks.iterrows():
            st.markdown(
                f"- **{row['Dimensión']}**: {row['Urgencia 0-100']:.1f}/100 "
                f"({row['Tipo'].lower()})"
            )

    st.markdown("### Hallazgos dinámicos")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"""
            **Burnout y desgaste laboral**  
            La relación BU–DL es **{corr_direction(rho_bu_dl)} {corr_strength(rho_bu_dl)}**
            ($\\rho$ = {rho_bu_dl:.2f}, p = {p_bu_dl:.4f}).  
            Esto indica que cuando el agotamiento aumenta, el desgaste laboral tiende a moverse en la misma dirección.
            """
        )

        st.info(
            f"""
            **Satisfacción e intención de retiro**  
            La relación SAT–IR es **{corr_direction(rho_sat_ir)} {corr_strength(rho_sat_ir)}**
            ($\\rho$ = {rho_sat_ir:.2f}, p = {p_sat_ir:.4f}).  
            Si la relación es negativa, significa que mayor satisfacción se asocia con menor intención de retiro.
            """
        )

    with col2:
        st.success(
            f"""
            **Compromiso del líder y satisfacción**  
            La relación CL–SAT es **{corr_direction(rho_cl_sat)} {corr_strength(rho_cl_sat)}**
            ($\\rho$ = {rho_cl_sat:.2f}, p = {p_cl_sat:.4f}).  
            Esto ayuda a evaluar si el liderazgo funciona como recurso protector.
            """
        )

        st.warning(
            f"""
            **Compromiso del líder y burnout**  
            La relación CL–BU es **{corr_direction(rho_cl_bu)} {corr_strength(rho_cl_bu)}**
            ($\\rho$ = {rho_cl_bu:.2f}, p = {p_cl_bu:.4f}).  
            Una relación negativa sugiere que mejores prácticas de liderazgo se asocian con menor agotamiento.
            """
        )

    if not top_resources.empty:
        st.markdown("### Recursos organizacionales que requieren atención")
        st.caption(
            "En recursos, una urgencia alta significa que el puntaje promedio del recurso es bajo. "
            "Por eso puede representar ausencia de protección organizacional."
        )

        st.dataframe(
            top_resources[["Código", "Dimensión", "Media", "Urgencia 0-100"]].round(2),
            width="stretch",
            hide_index=True,
        )


def plot_sample_profile(data: pd.DataFrame) -> None:
    st.subheader("1. ¿Quiénes componen la muestra?")

    st.markdown(
        """
        Esta primera parte describe el perfil de los trabajadores analizados. 
        La lectura del bienestar laboral depende del contexto: sector, cargo, modalidad y sexo pueden modificar 
        la forma en que se experimentan las demandas y los recursos laborales.
        """
    )

    cols = st.columns(2)
    for i, col in enumerate(filter_cols):
        if col in data.columns:
            freq = (
                data[col]
                .astype(str)
                .value_counts(dropna=False)
                .reset_index()
            )
            freq.columns = [col, "Frecuencia"]

            fig = px.bar(
                freq,
                x=col,
                y="Frecuencia",
                text="Frecuencia",
                title=f"Distribución por {col}",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_title=col, yaxis_title="Frecuencia")
            cols[i % 2].plotly_chart(fig, width="stretch")


def plot_risk_ranking(risk_df: pd.DataFrame) -> None:
    st.subheader("2. ¿Dónde está la mayor urgencia de intervención?")

    st.markdown(
        """
        Todas las dimensiones se transforman a una escala común de **0 a 100** para poder compararlas.
        En variables de riesgo, valores altos significan mayor riesgo. En variables de recurso, valores bajos 
        significan mayor urgencia porque indican menor protección organizacional.
        """
    )

    fig = px.bar(
        risk_df.sort_values("Urgencia 0-100"),
        x="Urgencia 0-100",
        y="Dimensión",
        color="Tipo",
        orientation="h",
        text="Urgencia 0-100",
        title="Ranking de urgencia psicosocial",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(xaxis_title="Urgencia 0-100", yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        risk_df.round(2),
        width="stretch",
        hide_index=True,
    )


def plot_dimension_distribution(data: pd.DataFrame) -> None:
    st.subheader("3. ¿Cómo se distribuyen las dimensiones principales?")

    st.markdown(
        """
        Los histogramas muestran la frecuencia de trabajadores en cada rango de puntaje.
        Esto permite identificar si una dimensión está concentrada en niveles bajos, medios o altos.
        """
    )

    selected_dim = st.selectbox(
        "Selecciona una dimensión para explorar su distribución",
        options=dimension_cols,
        format_func=lambda x: f"{x} - {dimensions[x]['label']}",
    )

    meta = dimensions[selected_dim]

    fig = px.histogram(
        data,
        x=selected_dim,
        nbins=20,
        marginal="box",
        title=f"Distribución de {selected_dim} - {meta['label']}",
    )
    fig.update_layout(
        xaxis_title="Puntaje promedio",
        yaxis_title="Frecuencia"
    )
    st.plotly_chart(fig, width="stretch")

    mean_value = data[selected_dim].mean()
    median_value = data[selected_dim].median()
    std_value = data[selected_dim].std()

    st.markdown(
        f"""
        **Interpretación:**  
        La media de **{meta['label']}** es **{mean_value:.2f}**, la mediana es **{median_value:.2f}**
        y la desviación estándar es **{std_value:.2f}**.  
        La media resume el nivel promedio de la dimensión, la mediana representa el valor central
        y la desviación estándar muestra qué tanta variabilidad existe entre trabajadores.
        """
    )


def plot_key_relationships(data: pd.DataFrame) -> None:
    st.subheader("4. ¿Qué variables se mueven juntas?")

    st.markdown(
        """
        Estas gráficas muestran relaciones clave del análisis bivariado. Se usa **Spearman** porque las dimensiones 
        provienen de escalas Likert y no siempre siguen una distribución normal. Spearman mide relaciones monótonas 
        y es más robusto para variables ordinales o con asimetrías.
        """
    )

    pair_options = {
        "Burnout vs Desgaste laboral": ("BU", "DL"),
        "Burnout vs Somatización": ("BU", "SOM"),
        "Desgaste laboral vs Somatización": ("DL", "SOM"),
        "Compromiso del líder vs Satisfacción": ("CL", "SAT"),
        "Compromiso del líder vs Burnout": ("CL", "BU"),
        "Satisfacción vs Intención de retiro": ("SAT", "IR"),
    }

    selected_pair = st.selectbox("Selecciona una relación clave", list(pair_options.keys()))
    x, y = pair_options[selected_pair]
    rho, p_value = corr_value(data, x, y)

    fig = px.scatter(
        data,
        x=x,
        y=y,
        color="Modalidad" if "Modalidad" in data.columns else None,
        trendline="ols",
        title=f"{selected_pair} | Spearman rho={rho:.2f}, p={p_value:.4f}",
        labels={
            x: f"{x} - {dimensions[x]['label']}",
            y: f"{y} - {dimensions[y]['label']}",
        },
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"""
        **Interpretación:**  
        La correlación entre **{dimensions[x]['label']}** y **{dimensions[y]['label']}** es 
        **{corr_direction(rho)} {corr_strength(rho)}**.  
        El valor p es **{p_value:.5f}**, por lo que la asociación 
        {'es estadísticamente significativa al 5%' if p_value < 0.05 else 'no es estadísticamente significativa al 5%'}.
        Este resultado describe asociación estadística, no causalidad directa.
        """
    )

    corr_matrix = data[dimension_cols].corr(method="spearman")

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title="Matriz de correlaciones Spearman entre dimensiones",
    )
    st.plotly_chart(fig_corr, width="stretch")


def plot_group_comparisons(data: pd.DataFrame) -> None:
    st.subheader("5. ¿Cambian los indicadores según cargo, sector o modalidad?")

    st.markdown(
        """
        Las comparaciones por grupo permiten identificar segmentos de trabajadores con mayor exposición a riesgo 
        o con mejores condiciones de bienestar.
        """
    )

    comparisons = {
        "Burnout por Tipo de Cargo": ("Tipo_Cargo", "BU"),
        "Satisfacción por Sector": ("Sector", "SAT"),
        "Desgaste por Modalidad": ("Modalidad", "DL"),
        "Presión del tiempo por Personas a Cargo": ("Personas_Cargo", "PT"),
    }

    selected = st.selectbox("Selecciona una comparación", list(comparisons.keys()))
    x, y = comparisons[selected]

    if x not in data.columns:
        st.warning(f"No se encontró la columna `{x}` en el dataset.")
        return

    fig = px.box(
        data,
        x=x,
        y=y,
        color=x,
        points="outliers",
        title=selected,
        labels={
            x: x,
            y: f"{y} - {dimensions[y]['label']}",
        },
    )
    st.plotly_chart(fig, width="stretch")

    group_table = (
        data.groupby(x, dropna=False)[y]
        .agg(["count", "mean", "median", "std"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )

    st.markdown("**Resumen por grupo**")
    st.dataframe(group_table.round(2), width="stretch", hide_index=True)

    top_group = group_table.iloc[0]
    st.markdown(
        f"""
        **Interpretación:**  
        El grupo con mayor promedio en **{dimensions[y]['label']}** es 
        **{top_group[x]}**, con una media de **{top_group['mean']:.2f}**.  
        Esta comparación permite priorizar segmentos específicos para intervención o seguimiento.
        """
    )


def plot_final_story(data: pd.DataFrame, risk_df: pd.DataFrame) -> None:
    st.subheader("6. Cierre: hallazgos y recomendaciones")

    if data.empty:
        return

    top_3 = risk_df.head(3)

    rho_sat_ir, _ = corr_value(data, "SAT", "IR")
    rho_bu_dl, _ = corr_value(data, "BU", "DL")
    rho_cl_sat, _ = corr_value(data, "CL", "SAT")

    st.markdown(
        f"""
        ### Hallazgo principal

        El análisis muestra que el bienestar laboral no depende de una sola variable, sino del equilibrio entre 
        **demandas laborales** y **recursos organizacionales**.

        En la muestra filtrada, las tres dimensiones que aparecen como mayor prioridad son:

        1. **{top_3.iloc[0]["Dimensión"]}** ({top_3.iloc[0]["Urgencia 0-100"]:.1f}/100)
        2. **{top_3.iloc[1]["Dimensión"]}** ({top_3.iloc[1]["Urgencia 0-100"]:.1f}/100)
        3. **{top_3.iloc[2]["Dimensión"]}** ({top_3.iloc[2]["Urgencia 0-100"]:.1f}/100)

        La relación entre **Burnout y Desgaste laboral** es {corr_direction(rho_bu_dl)} 
        {corr_strength(rho_bu_dl)} ($\\rho$ = {rho_bu_dl:.2f}), lo cual sugiere que el agotamiento emocional 
        y el deterioro laboral tienden a presentarse juntos.

        La relación entre **Satisfacción e Intención de retiro** es {corr_direction(rho_sat_ir)} 
        {corr_strength(rho_sat_ir)} ($\\rho$ = {rho_sat_ir:.2f}). Si es negativa, confirma que una mayor 
        satisfacción laboral se asocia con menor intención de abandonar la organización.

        Finalmente, la relación entre **Compromiso del líder y Satisfacción** es {corr_direction(rho_cl_sat)} 
        {corr_strength(rho_cl_sat)} ($\\rho$ = {rho_cl_sat:.2f}), lo que posiciona al liderazgo como un 
        posible recurso protector dentro del ambiente laboral.
        """
    )

    st.markdown("### Recomendaciones")
    st.markdown(
        """
        - Priorizar intervenciones sobre las dimensiones de mayor urgencia.
        - Fortalecer prácticas de liderazgo, acompañamiento y comunicación organizacional.
        - Monitorear burnout, desgaste y somatización como indicadores tempranos de riesgo.
        - Diseñar acciones diferenciadas por cargo, sector y modalidad de trabajo.
        - Usar este dashboard como herramienta de seguimiento y conversación con equipos de talento humano.
        """
    )


# ============================================================
# App
# ============================================================

df = load_data()
filtered = filtered_by_sidebar(df)

st.title("Dashboard — Bienestar laboral")
st.caption(
    "Análisis exploratorio de dimensiones psicosociales, demandas laborales, recursos organizacionales "
    "y patrones de bienestar en trabajadores."
)

if filtered.empty:
    st.warning("Los filtros seleccionados no tienen registros. Ajusta los filtros para continuar.")
    st.stop()

risk_df = build_risk_table(filtered)

# KPIs principales
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Trabajadores", len(filtered))
kpi2.metric("Burnout", f"{filtered['BU'].mean():.2f}")
kpi3.metric("Satisfacción", f"{filtered['SAT'].mean():.2f}")
kpi4.metric("Intención retiro", f"{filtered['IR'].mean():.2f}")
kpi5.metric("Mayor urgencia", risk_df.iloc[0]["Código"])

write_dynamic_story(filtered, risk_df)

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Perfil",
    "2. Riesgos",
    "3. Distribuciones",
    "4. Relaciones",
    "5. Grupos",
    "6. Cierre",
])

with tab1:
    plot_sample_profile(filtered)

with tab2:
    plot_risk_ranking(risk_df)

with tab3:
    plot_dimension_distribution(filtered)

with tab4:
    plot_key_relationships(filtered)

with tab5:
    plot_group_comparisons(filtered)

with tab6:
    plot_final_story(filtered, risk_df)
