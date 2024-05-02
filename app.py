import pandas as pd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import pandas as pd
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from sklearn.decomposition import PCA



class DataVisualizer():
    def __init__(self):
        self.PATH = './dataset'
        self.dataset_name = 'result_dataset.csv'
        self.df = pd.read_csv(self.PATH+"/"+self.dataset_name, delimiter=',')
        self.cluster_labels = None
        self.cluster_distributions = None
        self.transform_data()

    def __map_to_diet(self,x):
        if x == "meat":
            return "low_meat"
        if x == "meat50":
            return "med_mead"
        if x == "meat100":
            return "high_meat"
        if x == "veggie":
            return "vegetarian"
        else:
            return x
        
    def __map_age_groups(self,age):
        if age in ['20-29']:
            return 'young_adults'
        if age in ['30-39']:
            return 'adults'
        elif age in ['40-49']:
            return 'middle_aged'
        elif age in ['60-69','50-59', '70-79']:
            return 'seniors'

    def transform_data(self):
       self.df['diet_group'] = self.df['diet_group'].map(self.__map_to_diet)
       self.df['age_group'] = self.df['age_group'].map(self.__map_age_groups)

    def set_cluster_labels(self,labels):
        self.cluster_labels = labels
        self.cluster_distributions_dict = {}
        for i in range(len(labels)):
            self.cluster_distributions_dict[i] = labels[i]

    def generate_generic_labels(self):
        labels = []

        for cluster_id, dists in self.cluster_distributions.items():
            label = f"cluster {cluster_id}"

            labels.append(label)

        return labels


    def generate_cluster_labels(self):
        labels = []
        label_counts = {}

        for cluster_id, dists in self.cluster_distributions.items():
            top_diet = dists['Diet Distribution'].idxmax()
            top_diet_prop = dists['Diet Distribution'].max()

            top_gender = dists['Gender Distribution'].idxmax()
            top_gender_prop = dists['Gender Distribution'].max()

            top_age_group = dists['Age Distribution'].idxmax()
            top_age_group_prop = dists['Age Distribution'].max()

            label = f"{top_diet} Dominant ({top_diet_prop:.2f}), Largely {top_age_group} ({top_age_group_prop:.2f})"


            if label in label_counts:
                label_counts[label] += 1
                label += f" - Group {label_counts[label]}"
            else:
                label_counts[label] = 1

            labels.append(label)

        return labels
    



    def cluster_data(self,K,features):
        X = self.df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clustering = KMeans(n_clusters=K)
        clusters = clustering.fit_predict(X_scaled)

        self.df['cluster'] = clusters

        self.cluster_distributions = {}
        for i in range(K):
            cluster_data = self.df[self.df['cluster'] == i]
            diet_distribution = cluster_data['diet_group'].value_counts(normalize=True)
            gender_distribution = cluster_data['sex'].value_counts(normalize=True)
            age_distribution = cluster_data['age_group'].value_counts(normalize=True)
            self.cluster_distributions[i] = {
                'Diet Distribution': diet_distribution,
                'Gender Distribution': gender_distribution,
                'Age Distribution': age_distribution
            }

    def cluster_data_with_pca(self,K,features):
        X = self.df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        clustering = KMeans(n_clusters=K)
        clusters = clustering.fit_predict(X_pca)


        self.df['cluster'] = clusters

        self.cluster_distributions = {}
        for i in range(K):
            cluster_data = self.df[self.df['cluster'] == i]
            diet_distribution = cluster_data['diet_group'].value_counts(normalize=True)
            gender_distribution = cluster_data['sex'].value_counts(normalize=True)
            age_distribution = cluster_data['age_group'].value_counts(normalize=True)
            self.cluster_distributions[i] = {
                'Diet Distribution': diet_distribution,
                'Gender Distribution': gender_distribution,
                'Age Distribution': age_distribution
            }


    def plot_3D_scatter_plot(self):

        self.df['cluster_label'] = self.df['cluster'].map({i: label for i, label in enumerate(self.cluster_labels)})


        self.df['hover_text'] = self.df['cluster'].apply(lambda x: f"Cluster Label: {self.cluster_labels[x]}<br>")


        fig = px.scatter_3d(self.df,
                            x='mean_ghgs',  # X-axis - Mean greenhouse gas emissions
                            y='mean_watscar',  # Y-axis - Mean water scarcity
                            z='mean_acid',  # Z-axis - Mean acidification
                            color='cluster_label',  # Color by cluster label for readability
                            hover_name='cluster_label',  # Show cluster label on hover
                            hover_data=['hover_text'],  # Custom text with demographic info
                            labels={
                                'mean_ghgs': 'Mean GHG Emissions',
                                'mean_watscar': 'Mean Water Scarcity',
                                'mean_acid': 'Mean Acidification'
                            },)
        
        return fig

    def plot_treemap(self):

        rows = []
        for cluster_id, dists in self.cluster_distributions.items():
            for diet, diet_prop in dists['Diet Distribution'].items():
                for age, age_prop in dists['Age Distribution'].items():
                    rows.append({
                        'Cluster': f"{self.cluster_distributions_dict[cluster_id]}",
                        'Diet': diet,
                        'Age Group': age,
                        'Proportion': diet_prop * age_prop 
                    })

        df_treemap = pd.DataFrame(rows)

        fig = px.treemap(df_treemap, 
                        path=['Cluster', 'Diet', 'Age Group'],
                        values='Proportion',
                        color='Proportion',
                        color_continuous_scale='sunsetdark',
        )


        return fig
    
    def plot_parallel_coordinates(self):

        unique_labels = self.df['cluster_label'].unique()
        label_map = {label: i for i, label in enumerate(unique_labels)}
        self.df['color'] = self.df['cluster_label'].map(label_map)

        fig = px.parallel_coordinates(
            self.df,
            color='color', 
            dimensions=['mean_ghgs', 'mean_watscar', 'mean_acid'],
            labels={
                'mean_ghgs': 'GHG Emissions (kg CO2 eq)',
                'mean_watscar': 'Water Scarcity (cubic meters)',
                'mean_acid': 'Acidification (pH units)'
            },
            color_continuous_scale=px.colors.diverging.Tealrose,
            title='Parallel Coordinates Plot for Environmental Impacts'
        )

        fig.update_layout(
            coloraxis_colorbar=dict(
                title='Cluster Label',
                tickvals=list(label_map.values()),
                ticktext=list(label_map.keys())
            )
        )

        return fig



data_visualizer = DataVisualizer()


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# app.layout = html.Div([
#     dbc.Row([
#         dbc.Col([
#             dcc.Dropdown(
#                 id='cluster-number',
#                 options=[{'label': str(i), 'value': i} for i in range(2, 10)],
#                 value=3,
#                 className='mb-3'
#             )
#         ], width={'size': 6, 'offset': 3}),
#     ]),
#     dbc.Row([ 
#         dbc.Col(dcc.Graph(id='3d-scatter-plot'), width=6),
#         dbc.Col(dcc.Graph(id='parallel-coords-plot'), width=6),
#     ]),
#     dbc.Row([ 
#         dbc.Col(dcc.Graph(id='treemap-plot'), width=12)
#     ])
# ])

app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='cluster-number',
                options=[{'label': str(i), 'value': i} for i in range(2, 10)],
                value=3,  # default value
                className='mb-3'
            )
        ], width={'size': 4, 'offset': 4}),
    ]),
    dbc.Row([
        dbc.Col(dcc.Loading(
            id="loading-3d-scatter",
            children=[dcc.Graph(id='3d-scatter-plot')],
            type="default"),  # You can change the type to "circle", "cube", "dot", or "default"
            width=6),
        dbc.Col(dcc.Loading(
            id="loading-parallel-coords",
            children=[dcc.Graph(id='parallel-coords-plot')],
            type="default"),
            width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Loading(
            id="loading-treemap",
            children=[dcc.Graph(id='treemap-plot')],
            type="default"),
            width=12),
    ])
])


@app.callback(
    [Output('treemap-plot', 'figure'),
     Output('3d-scatter-plot', 'figure'),
     Output('parallel-coords-plot', 'figure')],
    [Input('cluster-number', 'value')]
)
def update_output(num_clusters):
    data_visualizer.cluster_data(num_clusters, [
    'mean_ghgs',  # Mean greenhouse gas emissions
    'mean_watscar',  # Mean water scarcity
    'mean_acid',  # Mean water use
    'mean_eut',
    'mean_ghgs_ch4',
    'mean_ghgs_n2o',
    'mean_land',
    'mean_watscar'
    ])
    # data_visualizer.cluster_data_with_pca(num_clusters,[
    # 'mean_ghgs',  # Mean greenhouse gas emissions
    # 'mean_watscar',  # Mean water scarcity
    # 'mean_acid',  # Mean water use
    # 'mean_eut',
    # 'mean_ghgs_ch4',
    # 'mean_ghgs_n2o',
    # 'mean_land',
    # 'mean_watscar'
    # ])
    labels = data_visualizer.generate_cluster_labels()
    data_visualizer.set_cluster_labels(labels)
    treemap_fig = data_visualizer.plot_treemap()
    scatter_fig = data_visualizer.plot_3D_scatter_plot()
    plot_parallel = data_visualizer.plot_parallel_coordinates()

    return treemap_fig, scatter_fig, plot_parallel

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)