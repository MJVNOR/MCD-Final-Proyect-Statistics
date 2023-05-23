import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from sklearn.utils import resample
import plotly.io as pio
import numpy as np
import warnings
import plotly.io as pio

pio.renderers.default = "browser"

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Proyecto para la materia de Estadistica",
    },
)

st.title("Trabajo Final Estadística")
st.header("Creación de un PCA")


@st.cache_data
def datos():
    bd1 = pd.read_excel("./Datos/Base_datos_1.xlsx")
    mybd1 = bd1.drop(["Tejido", "Replica", "Especie", "PESO "], axis=1)
    bd_V_farnesiana = bd1.loc[bd1["Especie"] == "V. farnesiana"].drop(
        ["Tejido", "Replica", "Especie", "PESO "], axis=1
    )
    bd_R_communis = bd1.loc[bd1["Especie"] == "R. communis"].drop(
        ["Tejido", "Replica", "Especie", "PESO "], axis=1
    )
    return bd1, bd_V_farnesiana, bd_R_communis, mybd1


bd1, bd_V_farnesiana, bd_R_communis, mybd1 = datos()

tab1, tab2, tab3, tab4 = st.tabs(["Datos", "Farnesiana", "Communis", "Todo"])

with tab1:
    st.subheader("Datos Originales")
    st.write("Base de datos de los minerales finales")
    st.dataframe(bd1)

    st.subheader("Farnesiana")
    st.write("Base de datos de la Farnesiana")
    st.dataframe(bd_R_communis)

    st.subheader("Communis")
    st.write("Base de datos de la Comunnis")
    st.dataframe(bd_R_communis)

array = ["Composta", "Composta&Inocuo", "Jal", "Jal&Inocuo", "SueloNatural"]


with tab2:
    option = st.selectbox(
        "Selecciona el tratamiento",
        ("C", "CI", "J", "JI", "SN"),
    )
    tratamiento = option
    st.subheader("Farnesiana PCA")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("PCA sin Boostrap")
        tratamientos = ["C", "CI", "J", "JI", "SN"]

        st.divider()
        indice = tratamientos.index(tratamiento)
        valor = array[indice]
        st.subheader(valor)
        st.divider()
        mi_db = (
            bd_V_farnesiana.loc[bd_V_farnesiana["Trat"] == tratamiento]
            .copy()
            .drop(["Trat", "Biomasa"], axis=1)
        )
        mi_db2 = (
            bd_V_farnesiana.loc[bd_V_farnesiana["Trat"] == tratamiento]
            .copy()
            .drop(["Trat"], axis=1)
        )
        pca_pipe = make_pipeline(StandardScaler(), PCA())

        # Se entrena y extrae el modelo entrenado del pipeline
        modelo_pca = pca_pipe.fit(mi_db).named_steps["pca"]

        # Se combierte el array a dataframe para añadir nombres a los ejes.
        componentes = []

        for componente in range(len(modelo_pca.components_)):
            nombre = "PC{}".format(componente + 1)
            componentes.append(nombre)

        pca_df = pd.DataFrame(
            data=modelo_pca.components_, columns=mi_db.columns, index=componentes
        )

        # print(np.transpose(modelo_pca.components_))
        # print(modelo_pca.components_)
        pca_df_componentes = pd.DataFrame(
            data=np.transpose(modelo_pca.components_),
            columns=componentes,
        )

        pca_df_variance = pd.DataFrame(
            data=[modelo_pca.explained_variance_ratio_],
            columns=componentes,
        )

        prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
        pca_df_acum = pd.DataFrame(
            data=[prop_varianza_acum],
            columns=componentes,
        )
        df_acumpc = pca_df_acum.T
        df_acumpc.rename(columns={0: "Value"}, inplace=True)

        # heatmap
        fig = px.imshow(pca_df)
        fig.update_layout(
            template="plotly_dark",
            title="Influencia de las variables en cada componente, Tratamiento: '%s'"
            % tratamiento,
        )
        fig.update_xaxes(title_text="Mineral")
        fig.update_yaxes(title_text="Componente")
        st.plotly_chart(fig, use_container_width=True)

        # Variaza y varianza acumulada
        df_variance_pca = pca_df_variance.T
        df_variance_pca.rename(columns={0: "Value"}, inplace=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_acumpc.index, y=df_acumpc["Value"], name="Varianza Acumulada"
            )
        )
        fig.add_trace(
            go.Bar(x=df_variance_pca.index, y=df_variance_pca["Value"], name="Varianza")
        )
        fig.update_xaxes(title_text="Componente")
        fig.update_yaxes(range=(0, 1.1), constrain="domain", title_text="Varianza")
        fig.update_layout(
            template="plotly_dark",
            title="Variaza y Varianza Acumulada, Tratamiento: '%s'" % tratamiento,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        proyecciones = pca_pipe.transform(X=mi_db)
        proyecciones = pd.DataFrame(
            proyecciones, columns=componentes, index=mi_db.index
        )

        if tratamiento == "J" or tratamiento == "JI":
            fig = px.scatter(
                proyecciones,
                x="PC1",
                y="PC2",
                # title=f'Total Explained Variance: {total_var:.2f}%',
                labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
            )
            fig.update_layout(
                template="plotly_dark",
                title="Visualización de dos componentes, Tratamiento: '%s'"
                % tratamiento,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter_3d(
                proyecciones,
                x="PC1",
                y="PC2",
                z="PC3",
                color=mi_db2["Biomasa"],
                # title=f'Total Explained Variance: {total_var:.2f}%',
                labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
            )
            fig.update_layout(
                template="plotly_dark",
                title="Visualización de tres componentes, Tratamiento: '%s'"
                % tratamiento,
            )
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("PCA con Boostrap")
        n_iterations = 1000
        new_data_df = pd.DataFrame()

        tratamientos = ["C", "CI", "J", "JI", "SN"]

        st.divider()
        indice = tratamientos.index(tratamiento)
        valor = array[indice]
        st.subheader(valor)
        st.divider()
        mi_db = (
            bd_V_farnesiana.loc[bd_V_farnesiana["Trat"] == tratamiento]
            .copy()
            .drop(["Trat"], axis=1)
        )
        new_data_df = pd.DataFrame()
        # Boostrap
        for i in range(n_iterations):
            newData = resample(mi_db, replace=True, n_samples=len(mi_db))
            new_data_df = new_data_df.append(
                newData.mean().to_frame().T, ignore_index=True
            )
        new_data_df = new_data_df.append(mi_db, ignore_index=True)

        # PCA
        mi_db = new_data_df.drop(["Biomasa"], axis=1)
        mi_db2 = new_data_df.copy()
        pca_pipe = make_pipeline(StandardScaler(), PCA())

        # Se entrena y extrae el modelo entrenado del pipeline
        modelo_pca = pca_pipe.fit(mi_db).named_steps["pca"]

        # Se combierte el array a dataframe para añadir nombres a los ejes.
        componentes = []

        for componente in range(len(modelo_pca.components_)):
            nombre = "PC{}".format(componente + 1)
            componentes.append(nombre)

        pca_df = pd.DataFrame(
            data=modelo_pca.components_, columns=mi_db.columns, index=componentes
        )

        # print(np.transpose(modelo_pca.components_))
        # print(modelo_pca.components_)
        pca_df_componentes = pd.DataFrame(
            data=np.transpose(modelo_pca.components_),
            columns=componentes,
        )

        pca_df_variance = pd.DataFrame(
            data=[modelo_pca.explained_variance_ratio_],
            columns=componentes,
        )

        prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
        pca_df_acum = pd.DataFrame(
            data=[prop_varianza_acum],
            columns=componentes,
        )
        df_acumpc = pca_df_acum.T
        df_acumpc.rename(columns={0: "Value"}, inplace=True)

        # heatmap
        fig = px.imshow(pca_df)
        fig.update_layout(
            template="plotly_dark",
            title="Influencia de las variables en cada componente, Tratamiento: '%s'"
            % tratamiento,
        )
        fig.update_xaxes(title_text="Mineral")
        fig.update_yaxes(title_text="Componente")
        st.plotly_chart(fig, use_container_width=True)

        # Variaza y varianza acumulada
        df_variance_pca = pca_df_variance.T
        df_variance_pca.rename(columns={0: "Value"}, inplace=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_acumpc.index, y=df_acumpc["Value"], name="Varianza Acumulada"
            )
        )
        fig.add_trace(
            go.Bar(x=df_variance_pca.index, y=df_variance_pca["Value"], name="Varianza")
        )
        fig.update_xaxes(title_text="Componente")
        fig.update_yaxes(range=(0, 1.1), constrain="domain", title_text="Varianza")
        fig.update_layout(
            template="plotly_dark",
            title="Variaza y Varianza Acumulada, Tratamiento: '%s'" % tratamiento,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        proyecciones = pca_pipe.transform(X=mi_db)
        proyecciones = pd.DataFrame(
            proyecciones, columns=componentes, index=mi_db.index
        )

        if tratamiento == "J" or tratamiento == "JI":
            fig = px.scatter(
                proyecciones,
                x="PC1",
                y="PC2",
                # title=f'Total Explained Variance: {total_var:.2f}%',
                labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
            )
            fig.update_layout(
                template="plotly_dark",
                title="Visualización de dos componentes, Tratamiento: '%s'"
                % tratamiento,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter_3d(
                proyecciones,
                x="PC1",
                y="PC2",
                z="PC3",
                color=mi_db2["Biomasa"],
                # title=f'Total Explained Variance: {total_var:.2f}%',
                labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
            )
            fig.update_layout(
                template="plotly_dark",
                title="Visualización de tres componentes, Tratamiento: '%s'"
                % tratamiento,
            )
            st.plotly_chart(fig, use_container_width=True)
with tab3:
    st.subheader("Communis PCA")
    option2 = st.selectbox(
        "Selecciona el tratamiento.",
        ("C", "CI", "J", "JI", "SN"),
    )
    tratamiento = option2
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("PCA sin Boostrap")
        tratamientos = ["C", "CI", "J", "JI", "SN"]

        st.divider()
        indice = tratamientos.index(tratamiento)
        valor = array[indice]
        st.subheader(valor)
        st.divider()
        mi_db = (
            bd_R_communis.loc[bd_R_communis["Trat"] == tratamiento]
            .copy()
            .drop(["Trat", "Biomasa"], axis=1)
        )
        mi_db2 = (
            bd_R_communis.loc[bd_R_communis["Trat"] == tratamiento]
            .copy()
            .drop(["Trat"], axis=1)
        )
        pca_pipe = make_pipeline(StandardScaler(), PCA())

        # Se entrena y extrae el modelo entrenado del pipeline
        modelo_pca = pca_pipe.fit(mi_db).named_steps["pca"]

        # Se combierte el array a dataframe para añadir nombres a los ejes.
        componentes = []

        for componente in range(len(modelo_pca.components_)):
            nombre = "PC{}".format(componente + 1)
            componentes.append(nombre)

        pca_df = pd.DataFrame(
            data=modelo_pca.components_, columns=mi_db.columns, index=componentes
        )

        # print(np.transpose(modelo_pca.components_))
        # print(modelo_pca.components_)
        pca_df_componentes = pd.DataFrame(
            data=np.transpose(modelo_pca.components_),
            columns=componentes,
        )

        pca_df_variance = pd.DataFrame(
            data=[modelo_pca.explained_variance_ratio_],
            columns=componentes,
        )

        prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
        pca_df_acum = pd.DataFrame(
            data=[prop_varianza_acum],
            columns=componentes,
        )
        df_acumpc = pca_df_acum.T
        df_acumpc.rename(columns={0: "Value"}, inplace=True)

        # heatmap
        fig = px.imshow(pca_df)
        fig.update_layout(
            template="plotly_dark",
            title="Influencia de las variables en cada componente, Tratamiento: '%s'"
            % tratamiento,
        )
        fig.update_xaxes(title_text="Mineral")
        fig.update_yaxes(title_text="Componente")
        st.plotly_chart(fig, use_container_width=True)

        # Variaza y varianza acumulada
        df_variance_pca = pca_df_variance.T
        df_variance_pca.rename(columns={0: "Value"}, inplace=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_acumpc.index, y=df_acumpc["Value"], name="Varianza Acumulada"
            )
        )
        fig.add_trace(
            go.Bar(x=df_variance_pca.index, y=df_variance_pca["Value"], name="Varianza")
        )
        fig.update_xaxes(title_text="Componente")
        fig.update_yaxes(range=(0, 1.1), constrain="domain", title_text="Varianza")
        fig.update_layout(
            template="plotly_dark",
            title="Variaza y Varianza Acumulada, Tratamiento: '%s'" % tratamiento,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        proyecciones = pca_pipe.transform(X=mi_db)
        proyecciones = pd.DataFrame(
            proyecciones, columns=componentes, index=mi_db.index
        )

        if tratamiento == "J" or tratamiento == "JI":
            fig = px.scatter(
                proyecciones,
                x="PC1",
                y="PC2",
                # title=f'Total Explained Variance: {total_var:.2f}%',
                labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
            )
            fig.update_layout(
                template="plotly_dark",
                title="Visualización de dos componentes, Tratamiento: '%s'"
                % tratamiento,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter_3d(
                proyecciones,
                x="PC1",
                y="PC2",
                z="PC3",
                color=mi_db2["Biomasa"],
                # title=f'Total Explained Variance: {total_var:.2f}%',
                labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
            )
            fig.update_layout(
                template="plotly_dark",
                title="Visualización de tres componentes, Tratamiento: '%s'"
                % tratamiento,
            )
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("PCA con Boostrap")
        n_iterations = 1000
        new_data_df = pd.DataFrame()

        tratamientos = ["C", "CI", "J", "JI", "SN"]

        st.divider()
        indice = tratamientos.index(tratamiento)
        valor = array[indice]
        st.subheader(valor)
        st.divider()
        mi_db = (
            bd_R_communis.loc[bd_R_communis["Trat"] == tratamiento]
            .copy()
            .drop(["Trat"], axis=1)
        )
        new_data_df = pd.DataFrame()
        # Boostrap
        for i in range(n_iterations):
            newData = resample(mi_db, replace=True, n_samples=len(mi_db))
            new_data_df = new_data_df.append(
                newData.mean().to_frame().T, ignore_index=True
            )
        new_data_df = new_data_df.append(mi_db, ignore_index=True)

        # PCA
        mi_db = new_data_df.drop(["Biomasa"], axis=1)
        mi_db2 = new_data_df.copy()
        pca_pipe = make_pipeline(StandardScaler(), PCA())

        # Se entrena y extrae el modelo entrenado del pipeline
        modelo_pca = pca_pipe.fit(mi_db).named_steps["pca"]

        # Se combierte el array a dataframe para añadir nombres a los ejes.
        componentes = []

        for componente in range(len(modelo_pca.components_)):
            nombre = "PC{}".format(componente + 1)
            componentes.append(nombre)

        pca_df = pd.DataFrame(
            data=modelo_pca.components_, columns=mi_db.columns, index=componentes
        )

        # print(np.transpose(modelo_pca.components_))
        # print(modelo_pca.components_)
        pca_df_componentes = pd.DataFrame(
            data=np.transpose(modelo_pca.components_),
            columns=componentes,
        )

        pca_df_variance = pd.DataFrame(
            data=[modelo_pca.explained_variance_ratio_],
            columns=componentes,
        )

        prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
        pca_df_acum = pd.DataFrame(
            data=[prop_varianza_acum],
            columns=componentes,
        )
        df_acumpc = pca_df_acum.T
        df_acumpc.rename(columns={0: "Value"}, inplace=True)

        # heatmap
        fig = px.imshow(pca_df)
        fig.update_layout(
            template="plotly_dark",
            title="Influencia de las variables en cada componente, Tratamiento: '%s'"
            % tratamiento,
        )
        fig.update_xaxes(title_text="Mineral")
        fig.update_yaxes(title_text="Componente")
        st.plotly_chart(fig, use_container_width=True)

        # Variaza y varianza acumulada
        df_variance_pca = pca_df_variance.T
        df_variance_pca.rename(columns={0: "Value"}, inplace=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_acumpc.index, y=df_acumpc["Value"], name="Varianza Acumulada"
            )
        )
        fig.add_trace(
            go.Bar(x=df_variance_pca.index, y=df_variance_pca["Value"], name="Varianza")
        )
        fig.update_xaxes(title_text="Componente")
        fig.update_yaxes(range=(0, 1.1), constrain="domain", title_text="Varianza")
        fig.update_layout(
            template="plotly_dark",
            title="Variaza y Varianza Acumulada, Tratamiento: '%s'" % tratamiento,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        proyecciones = pca_pipe.transform(X=mi_db)
        proyecciones = pd.DataFrame(
            proyecciones, columns=componentes, index=mi_db.index
        )

        if tratamiento == "J" or tratamiento == "JI":
            fig = px.scatter(
                proyecciones,
                x="PC1",
                y="PC2",
                # title=f'Total Explained Variance: {total_var:.2f}%',
                labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
            )
            fig.update_layout(
                template="plotly_dark",
                title="Visualización de dos componentes, Tratamiento: '%s'"
                % tratamiento,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter_3d(
                proyecciones,
                x="PC1",
                y="PC2",
                z="PC3",
                color=mi_db2["Biomasa"],
                # title=f'Total Explained Variance: {total_var:.2f}%',
                labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
            )
            fig.update_layout(
                template="plotly_dark",
                title="Visualización de tres componentes, Tratamiento: '%s'"
                % tratamiento,
            )
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Todo")
    st.subheader("Communis PCA")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("PCA sin Boostrap")

        mi_db = mybd1.drop(["Trat", "Biomasa"], axis=1)
        mi_db2 = mybd1.drop(["Trat"], axis=1)
        pca_pipe = make_pipeline(StandardScaler(), PCA())

        # Se entrena y extrae el modelo entrenado del pipeline
        modelo_pca = pca_pipe.fit(mi_db).named_steps["pca"]

        # Se combierte el array a dataframe para añadir nombres a los ejes.
        componentes = []

        for componente in range(len(modelo_pca.components_)):
            nombre = "PC{}".format(componente + 1)
            componentes.append(nombre)

        pca_df = pd.DataFrame(
            data=modelo_pca.components_, columns=mi_db.columns, index=componentes
        )

        # print(np.transpose(modelo_pca.components_))
        # print(modelo_pca.components_)
        pca_df_componentes = pd.DataFrame(
            data=np.transpose(modelo_pca.components_),
            columns=componentes,
        )

        pca_df_variance = pd.DataFrame(
            data=[modelo_pca.explained_variance_ratio_],
            columns=componentes,
        )

        prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
        pca_df_acum = pd.DataFrame(
            data=[prop_varianza_acum],
            columns=componentes,
        )
        df_acumpc = pca_df_acum.T
        df_acumpc.rename(columns={0: "Value"}, inplace=True)

        # heatmap
        fig = px.imshow(pca_df)
        fig.update_layout(
            template="plotly_dark",
            title="Influencia de las variables en cada componente, Tratamiento: '%s'"
            % tratamiento,
        )
        fig.update_xaxes(title_text="Mineral")
        fig.update_yaxes(title_text="Componente")
        st.plotly_chart(fig, use_container_width=True)

        # Variaza y varianza acumulada
        df_variance_pca = pca_df_variance.T
        df_variance_pca.rename(columns={0: "Value"}, inplace=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_acumpc.index, y=df_acumpc["Value"], name="Varianza Acumulada"
            )
        )
        fig.add_trace(
            go.Bar(x=df_variance_pca.index, y=df_variance_pca["Value"], name="Varianza")
        )
        fig.update_xaxes(title_text="Componente")
        fig.update_yaxes(range=(0, 1.1), constrain="domain", title_text="Varianza")
        fig.update_layout(
            template="plotly_dark",
            title="Variaza y Varianza Acumulada, Tratamiento: '%s'" % tratamiento,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        proyecciones = pca_pipe.transform(X=mi_db)
        proyecciones = pd.DataFrame(
            proyecciones, columns=componentes, index=mi_db.index
        )

        if tratamiento == "J" or tratamiento == "JI":
            fig = px.scatter(
                proyecciones,
                x="PC1",
                y="PC2",
                # title=f'Total Explained Variance: {total_var:.2f}%',
                labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
            )
            fig.update_layout(
                template="plotly_dark",
                title="Visualización de dos componentes, Tratamiento: '%s'"
                % tratamiento,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter_3d(
                proyecciones,
                x="PC1",
                y="PC2",
                z="PC3",
                color=mi_db2["Biomasa"],
                # title=f'Total Explained Variance: {total_var:.2f}%',
                labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
            )
            fig.update_layout(
                template="plotly_dark",
                title="Visualización de tres componentes, Tratamiento: '%s'"
                % tratamiento,
            )
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("PCA con Boostrap")
        n_iterations = 1000
        new_data_df = pd.DataFrame()

        mi_db = mybd1.drop(["Trat"], axis=1)
        new_data_df = pd.DataFrame()
        # Boostrap
        for i in range(n_iterations):
            newData = resample(mi_db, replace=True, n_samples=len(mi_db))
            new_data_df = new_data_df.append(
                newData.mean().to_frame().T, ignore_index=True
            )
        new_data_df = new_data_df.append(mi_db, ignore_index=True)

        # PCA
        mi_db = new_data_df.drop(["Biomasa"], axis=1)
        mi_db2 = new_data_df.copy()
        pca_pipe = make_pipeline(StandardScaler(), PCA())

        # Se entrena y extrae el modelo entrenado del pipeline
        modelo_pca = pca_pipe.fit(mi_db).named_steps["pca"]

        # Se combierte el array a dataframe para añadir nombres a los ejes.
        componentes = []

        for componente in range(len(modelo_pca.components_)):
            nombre = "PC{}".format(componente + 1)
            componentes.append(nombre)

        pca_df = pd.DataFrame(
            data=modelo_pca.components_, columns=mi_db.columns, index=componentes
        )

        # print(np.transpose(modelo_pca.components_))
        # print(modelo_pca.components_)
        pca_df_componentes = pd.DataFrame(
            data=np.transpose(modelo_pca.components_),
            columns=componentes,
        )

        pca_df_variance = pd.DataFrame(
            data=[modelo_pca.explained_variance_ratio_],
            columns=componentes,
        )

        prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
        pca_df_acum = pd.DataFrame(
            data=[prop_varianza_acum],
            columns=componentes,
        )
        df_acumpc = pca_df_acum.T
        df_acumpc.rename(columns={0: "Value"}, inplace=True)

        # heatmap
        fig = px.imshow(pca_df)
        fig.update_layout(
            template="plotly_dark",
            title="Influencia de las variables en cada componente, Tratamiento: '%s'"
            % tratamiento,
        )
        fig.update_xaxes(title_text="Mineral")
        fig.update_yaxes(title_text="Componente")
        st.plotly_chart(fig, use_container_width=True)

        # Variaza y varianza acumulada
        df_variance_pca = pca_df_variance.T
        df_variance_pca.rename(columns={0: "Value"}, inplace=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_acumpc.index, y=df_acumpc["Value"], name="Varianza Acumulada"
            )
        )
        fig.add_trace(
            go.Bar(x=df_variance_pca.index, y=df_variance_pca["Value"], name="Varianza")
        )
        fig.update_xaxes(title_text="Componente")
        fig.update_yaxes(range=(0, 1.1), constrain="domain", title_text="Varianza")
        fig.update_layout(
            template="plotly_dark",
            title="Variaza y Varianza Acumulada, Tratamiento: '%s'" % tratamiento,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        proyecciones = pca_pipe.transform(X=mi_db)
        proyecciones = pd.DataFrame(
            proyecciones, columns=componentes, index=mi_db.index
        )

        if tratamiento == "J" or tratamiento == "JI":
            fig = px.scatter(
                proyecciones,
                x="PC1",
                y="PC2",
                # title=f'Total Explained Variance: {total_var:.2f}%',
                labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
            )
            fig.update_layout(
                template="plotly_dark",
                title="Visualización de dos componentes, Tratamiento: '%s'"
                % tratamiento,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter_3d(
                proyecciones,
                x="PC1",
                y="PC2",
                z="PC3",
                color=mi_db2["Biomasa"],
                # title=f'Total Explained Variance: {total_var:.2f}%',
                labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
            )
            fig.update_layout(
                template="plotly_dark",
                title="Visualización de tres componentes, Tratamiento: '%s'"
                % tratamiento,
            )
            st.plotly_chart(fig, use_container_width=True)
