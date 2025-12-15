#!/usr/bin/env python3
"""
LLM Benchmark Visualization Dashboard
======================================
Interactive visualization of LLM evaluation results using Plotly Dash.
Supports multi-file analysis with filtering and comparison charts.
"""

import json
import os
from pathlib import Path
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import base64

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
EVALUATIONS_DIR = PROJECT_ROOT / "evaluations"

# Color scheme
COLORS = {
    'background': '#0f0f1a',
    'card': '#1a1a2e',
    'text': '#ffffff',
    'text_secondary': '#a0a0b0',
    'accent': '#4ecdc4',
    'purple': '#a855f7',
    'green': '#22c55e',
    'yellow': '#eab308',
    'red': '#ef4444',
    'blue': '#3b82f6',
    'orange': '#f97316',
    'pink': '#ec4899'
}

# Category colors
CATEGORY_COLORS = [
    '#4ecdc4', '#a855f7', '#ec4899', '#22c55e', '#eab308',
    '#ef4444', '#3b82f6', '#f97316', '#06b6d4', '#8b5cf6'
]

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)

app.title = "LLM Benchmark Dashboard"

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #0f0f1a;
            }
            .dash-graph {
                background-color: #1a1a2e;
                border-radius: 12px;
                padding: 15px;
            }
            .card {
                background-color: #1a1a2e !important;
                border: 1px solid #3f3f5a;
                border-radius: 12px;
            }
            .upload-box {
                border: 2px dashed #3f3f5a;
                border-radius: 12px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
            }
            .upload-box:hover {
                border-color: #4ecdc4;
                background: rgba(78, 205, 196, 0.05);
            }
            .filter-card {
                background: #252540;
                border-radius: 8px;
                padding: 15px;
            }
            .stat-value {
                font-size: 2.5rem;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def create_layout():
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🧠 LLM Benchmark Dashboard", 
                       className="text-center mb-2",
                       style={'background': 'linear-gradient(135deg, #4ecdc4, #a855f7)',
                              '-webkit-background-clip': 'text',
                              '-webkit-text-fill-color': 'transparent',
                              'fontSize': '2.5rem'}),
                html.P("Upload evaluation JSON files to visualize results", 
                       className="text-center text-muted mb-4")
            ])
        ], className="mt-4"),
        
        # File Upload Section
        dbc.Row([
            dbc.Col([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        html.H4("📂 Drop evaluation JSON files here"),
                        html.P("or click to browse", className="text-muted"),
                        html.P("Supports multiple files for comparison", className="text-muted small")
                    ]),
                    className="upload-box",
                    multiple=True
                ),
                html.Div(id='file-list', className="mt-3")
            ], width=12)
        ], className="mb-4"),
        
        # Filters Section (hidden until files loaded)
        html.Div(id='filters-section', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("🔧 Filters", className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Score Range", className="mb-2"),
                                    dcc.RangeSlider(
                                        id='score-filter',
                                        min=0, max=10, step=1,
                                        value=[0, 10],
                                        marks={i: str(i) for i in range(11)},
                                        tooltip={"placement": "bottom"}
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("Quick Filters", className="mb-2"),
                                    dbc.ButtonGroup([
                                        dbc.Button("Low (<3)", id="filter-low", color="danger", size="sm", outline=True),
                                        dbc.Button("Mid (3-7)", id="filter-mid", color="warning", size="sm", outline=True),
                                        dbc.Button("High (8+)", id="filter-high", color="success", size="sm", outline=True),
                                        dbc.Button("Perfect (10)", id="filter-perfect", color="info", size="sm", outline=True),
                                        dbc.Button("All", id="filter-all", color="secondary", size="sm", outline=True),
                                    ], className="flex-wrap")
                                ], md=6)
                            ])
                        ])
                    ], className="mb-4")
                ])
            ])
        ], style={'display': 'none'}),
        
        # Stats Cards
        html.Div(id='stats-section', style={'display': 'none'}, children=[
            dbc.Row(id='stats-row', className="mb-4")
        ]),
        
        # Charts Section
        html.Div(id='charts-section', style={'display': 'none'}, children=[
            # Chart Type Selector
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("📊 Select Visualization", className="mb-3"),
                            dbc.Tabs(id='chart-tabs', active_tab='scatter', children=[
                                dbc.Tab(label="Score vs Response Time", tab_id="scatter"),
                                dbc.Tab(label="Category Comparison", tab_id="category"),
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Main Chart Area
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='main-chart', style={'height': '500px'})
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Secondary Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("📈 Score Distribution", className="mb-3"),
                            dcc.Graph(id='distribution-chart', style={'height': '350px'})
                        ])
                    ])
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("🎯 Model Comparison", className="mb-3"),
                            dcc.Graph(id='comparison-chart', style={'height': '350px'})
                        ])
                    ])
                ], md=6)
            ])
        ]),
        
        # Data store
        dcc.Store(id='evaluation-data', data=[]),
        
    ], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '20px'})


app.layout = create_layout


def parse_evaluation_file(contents, filename):
    """Parse uploaded evaluation JSON file."""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        data = json.loads(decoded.decode('utf-8'))
        
        # Extract evaluations with metadata
        evaluations = []
        model = data.get('model', 'unknown')
        dataset = data.get('dataset', 'unknown')
        
        for ev in data.get('evaluations', []):
            if ev.get('score') is not None:
                evaluations.append({
                    'question_id': ev.get('question_id', ''),
                    'prompt': ev.get('prompt', '')[:100],
                    'response': ev.get('response', '')[:200],
                    'score': ev.get('score', 0),
                    'response_time': ev.get('response_time', 0),
                    'domain': ev.get('domain', 'Unknown'),
                    'category': ev.get('category', 'Unknown'),
                    'remark': ev.get('remark', ''),
                    'model': model,
                    'dataset': dataset,
                    'source_file': filename
                })
        
        return evaluations, None
    except Exception as e:
        return None, str(e)


@callback(
    [Output('evaluation-data', 'data'),
     Output('file-list', 'children'),
     Output('filters-section', 'style'),
     Output('stats-section', 'style'),
     Output('charts-section', 'style')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents_list, filenames):
    if not contents_list:
        return [], "", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    
    all_evaluations = []
    file_badges = []
    
    for contents, filename in zip(contents_list, filenames):
        evaluations, error = parse_evaluation_file(contents, filename)
        if evaluations:
            all_evaluations.extend(evaluations)
            file_badges.append(
                dbc.Badge(f"✓ {filename}", color="success", className="me-2 mb-2")
            )
        else:
            file_badges.append(
                dbc.Badge(f"✗ {filename}: {error}", color="danger", className="me-2 mb-2")
            )
    
    show = {'display': 'block'} if all_evaluations else {'display': 'none'}
    
    return all_evaluations, html.Div(file_badges), show, show, show


@callback(
    Output('stats-row', 'children'),
    Input('evaluation-data', 'data'),
    Input('score-filter', 'value')
)
def update_stats(data, score_range):
    if not data:
        return []
    
    df = pd.DataFrame(data)
    df = df[(df['score'] >= score_range[0]) & (df['score'] <= score_range[1])]
    
    if df.empty:
        return [dbc.Col(html.P("No data matching filters", className="text-muted text-center"))]
    
    total = len(df)
    avg_score = df['score'].mean()
    avg_time = df['response_time'].mean()
    models = df['model'].nunique()
    high_scores = len(df[df['score'] >= 8])
    low_scores = len(df[df['score'] < 3])
    
    stats = [
        ("Total Evaluated", total, "primary"),
        ("Avg Score", f"{avg_score:.1f}/10", "success" if avg_score >= 7 else "warning" if avg_score >= 4 else "danger"),
        ("Avg Response Time", f"{avg_time:.2f}s", "info"),
        ("Models", models, "secondary"),
        ("High Scores (8+)", high_scores, "success"),
        ("Low Scores (<3)", low_scores, "danger")
    ]
    
    return [
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(value, className=f"stat-value text-{color}"),
                    html.P(label, className="text-muted mb-0")
                ], className="text-center")
            ])
        ], md=2) for label, value, color in stats
    ]


@callback(
    Output('score-filter', 'value'),
    [Input('filter-low', 'n_clicks'),
     Input('filter-mid', 'n_clicks'),
     Input('filter-high', 'n_clicks'),
     Input('filter-perfect', 'n_clicks'),
     Input('filter-all', 'n_clicks')],
    prevent_initial_call=True
)
def quick_filter(low, mid, high, perfect, all_btn):
    from dash import ctx
    if not ctx.triggered:
        return [0, 10]
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'filter-low':
        return [0, 2]
    elif button_id == 'filter-mid':
        return [3, 7]
    elif button_id == 'filter-high':
        return [8, 10]
    elif button_id == 'filter-perfect':
        return [10, 10]
    else:
        return [0, 10]


@callback(
    Output('main-chart', 'figure'),
    [Input('evaluation-data', 'data'),
     Input('score-filter', 'value'),
     Input('chart-tabs', 'active_tab')]
)
def update_main_chart(data, score_range, active_tab):
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    df = df[(df['score'] >= score_range[0]) & (df['score'] <= score_range[1])]
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data matching filters", 
                          xref="paper", yref="paper", x=0.5, y=0.5, 
                          showarrow=False, font=dict(size=20, color=COLORS['text_secondary']))
        fig.update_layout(template='plotly_dark', paper_bgcolor=COLORS['card'], plot_bgcolor=COLORS['card'])
        return fig
    
    if active_tab == 'scatter':
        # Scatter plot: X = Score, Y = Response Time
        fig = px.scatter(
            df,
            x='score',
            y='response_time',
            color='category',
            symbol='model',
            size=[10] * len(df),
            hover_data=['question_id', 'remark', 'domain'],
            color_discrete_sequence=CATEGORY_COLORS,
            labels={'score': 'Score (0-10)', 'response_time': 'Response Time (seconds)'}
        )
        fig.update_layout(
            title='Score vs Response Time',
            xaxis=dict(range=[-0.5, 10.5], dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
    else:  # category
        # Category bar chart
        category_stats = df.groupby('category').agg({
            'score': 'mean',
            'response_time': 'mean',
            'question_id': 'count'
        }).reset_index()
        category_stats.columns = ['Category', 'Avg Score', 'Avg Time', 'Count']
        category_stats = category_stats.sort_values('Avg Score', ascending=True)
        
        fig = px.bar(
            category_stats,
            x='Avg Score',
            y='Category',
            orientation='h',
            color='Avg Score',
            color_continuous_scale=['#ef4444', '#eab308', '#22c55e'],
            range_color=[0, 10],
            hover_data=['Avg Time', 'Count'],
            text='Avg Score'
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(
            title='Average Score by Category',
            xaxis=dict(range=[0, 11]),
            coloraxis_showscale=False
        )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


@callback(
    Output('distribution-chart', 'figure'),
    [Input('evaluation-data', 'data'),
     Input('score-filter', 'value')]
)
def update_distribution(data, score_range):
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    df = df[(df['score'] >= score_range[0]) & (df['score'] <= score_range[1])]
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template='plotly_dark', paper_bgcolor=COLORS['card'], plot_bgcolor=COLORS['card'])
        return fig
    
    # Score distribution histogram
    fig = px.histogram(
        df,
        x='score',
        nbins=11,
        color='model',
        barmode='overlay',
        opacity=0.7,
        color_discrete_sequence=CATEGORY_COLORS
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis=dict(dtick=1, range=[-0.5, 10.5]),
        bargap=0.1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


@callback(
    Output('comparison-chart', 'figure'),
    [Input('evaluation-data', 'data'),
     Input('score-filter', 'value')]
)
def update_comparison(data, score_range):
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    df = df[(df['score'] >= score_range[0]) & (df['score'] <= score_range[1])]
    
    if df.empty or df['model'].nunique() < 1:
        fig = go.Figure()
        fig.update_layout(template='plotly_dark', paper_bgcolor=COLORS['card'], plot_bgcolor=COLORS['card'])
        return fig
    
    # Model comparison
    model_stats = df.groupby('model').agg({
        'score': ['mean', 'std', 'count'],
        'response_time': 'mean'
    }).reset_index()
    model_stats.columns = ['Model', 'Avg Score', 'Std Dev', 'Count', 'Avg Time']
    model_stats = model_stats.sort_values('Avg Score', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=model_stats['Model'],
        y=model_stats['Avg Score'],
        marker_color=[CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i in range(len(model_stats))],
        text=model_stats['Avg Score'].apply(lambda x: f'{x:.1f}'),
        textposition='outside',
        error_y=dict(type='data', array=model_stats['Std Dev'], visible=True)
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        yaxis=dict(range=[0, 11]),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def main():
    print(f"""
\033[96m
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   🧠 LLM BENCHMARK VISUALIZATION DASHBOARD                        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
\033[0m
  Starting server on: \033[92mhttp://localhost:8050\033[0m
  
  Press Ctrl+C to stop the server.
""")
    
    app.run(debug=False, host='0.0.0.0', port=8050)


if __name__ == "__main__":
    main()
