from dash import Dash, html
import dash_cytoscape as cyto

app = Dash(__name__)

elements = [
    # nodes
    {'data': {'id': 'A', 'label': 'Alice'}},
    {'data': {'id': 'B', 'label': 'Bob'}},
    {'data': {'id': 'C', 'label': 'Carol'}},

    # edges
    {'data': {'source': 'A', 'target': 'B', 'label': 'knows'}},
    {'data': {'source': 'A', 'target': 'C', 'label': 'works with'}},
    {'data': {'source': 'B', 'target': 'C', 'label': 'reports to'}},
]

app.layout = html.Div([
    cyto.Cytoscape(
        id='graph',
        elements=elements,
        layout={'name': 'cose'},   # auto layout
        style={'width': '100%', 'height': '700px'},
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'label': 'data(label)',
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle'
                }
            }
        ]
    )
])

if __name__ == '__main__':
    app.run(debug=True)