"""
Theme utilities for consistent visualization styling across the app.
"""

THEME = {
    'background': '#111111',
    'text': '#FFFFFF',
    'grid': '#333333',
    'primary': '#00B4D8',
    'secondary': '#90E0EF',
    'accent': '#CAF0F8',
    'warning': '#FF9E00',
    'success': '#2ECC71',
    'danger': '#E74C3C'
}

def apply_theme_to_plotly(fig):
    """Apply consistent theme to plotly figures."""
    fig.update_layout(
        plot_bgcolor=THEME['background'],
        paper_bgcolor=THEME['background'],
        font_color=THEME['text'],
        title_font_color=THEME['text'],
        title_x=0.5,  
        title_xanchor='center',  
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        xaxis=dict(
            gridcolor=THEME['grid'],
            zerolinecolor=THEME['grid']
        ),
        yaxis=dict(
            gridcolor=THEME['grid'],
            zerolinecolor=THEME['grid']
        )
    )
    return fig

def get_streamlit_theme():
    """Get Streamlit CSS for consistent theme."""
    return """
    <style>
    /* Main app styles */
    .stApp {
        background-color: """ + THEME['background'] + """;
        color: """ + THEME['text'] + """;
    }
    
    /* Navigation Menu Styling */
    .nav-header {
        text-align: center;
        padding: 20px 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 25px;
        background: linear-gradient(to right, #f8f9fa, #ffffff);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .nav-header h1 {
        font-size: 26px;
        margin-bottom: 8px;
        color: #000000;
        font-weight: 600;
    }
    
    .nav-header h3 {
        font-size: 16px;
        color: #555;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    .nav-section-header {
        font-size: 18px;
        font-weight: 600;
        color: #000000;
        margin: 20px 0 15px;
        padding: 10px 15px;
        background: linear-gradient(to right, #f8f9fa, #ffffff);
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .nav-item {
        padding: 15px;
        margin: 12px 0;
        border-radius: 12px;
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        border: 1px solid #f0f0f0;
    }
    
    .nav-item:hover {
        transform: translateX(8px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        border-color: #e0e0e0;
        background: linear-gradient(145deg, #ffffff, #f0f0f0);
    }
    
    .nav-item-title {
        font-size: 17px;
        font-weight: 600;
        color: #000000;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .nav-item-desc {
        font-size: 14px;
        color: #444;
        margin-bottom: 6px;
        font-weight: 500;
        padding-left: 2px;
    }
    
    .nav-item-detail {
        font-size: 12px;
        color: #666;
        font-style: italic;
        padding-left: 2px;
        line-height: 1.4;
    }
    
    .quick-guide {
        margin-top: 25px;
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    
    .guide-item {
        padding: 10px 12px;
        border-bottom: 1px solid #eee;
        transition: all 0.3s ease;
    }
    
    .guide-item:last-child {
        border-bottom: none;
    }
    
    .guide-item:hover {
        background-color: #f8f9fa;
        transform: translateX(5px);
    }
    
    .guide-title {
        font-size: 15px;
        font-weight: 600;
        color: #000000;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .guide-desc {
        font-size: 13px;
        color: #555;
        padding-left: 2px;
        line-height: 1.4;
    }
    
    /* Custom scrollbar for sidebar */
    .sidebar::-webkit-scrollbar {
        width: 6px;
    }
    
    .sidebar::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .sidebar::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }
    
    .sidebar::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Main content text */
    .main .stMarkdown, .main .stText, .main p, .main span {
        color: """ + THEME['text'] + """ !important;
    }
    
    /* Dropdown styles */
    .stSelectbox > div > div {
        background-color: """ + THEME['grid'] + """;
    }
    .stSelectbox [data-baseweb="select"] {
        color: """ + THEME['text'] + """ !important;
    }
    .stSelectbox [data-baseweb="select"] * {
        color: """ + THEME['text'] + """ !important;
    }
    .stSelectbox [role="listbox"] * {
        color: """ + THEME['text'] + """ !important;
        background-color: """ + THEME['grid'] + """ !important;
    }
    .stMultiSelect [data-baseweb="select"] {
        color: """ + THEME['text'] + """ !important;
    }
    .stMultiSelect [data-baseweb="select"] * {
        color: """ + THEME['text'] + """ !important;
    }
    .stMultiSelect [role="listbox"] * {
        color: """ + THEME['text'] + """ !important;
        background-color: """ + THEME['grid'] + """ !important;
    }
    
    /* Other UI elements */
    .stSlider > div > div {
        background-color: """ + THEME['grid'] + """;
    }
    div[data-testid="stMetricValue"] > div {
        color: """ + THEME['text'] + """ !important;
    }
    div[data-testid="stMetricLabel"] > label {
        color: """ + THEME['text'] + """ !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: """ + THEME['grid'] + """;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: """ + THEME['background'] + """;
        border-radius: 4px 4px 0px 0px;
        color: """ + THEME['text'] + """;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: """ + THEME['primary'] + """;
    }
    button[kind="primary"] {
        background-color: """ + THEME['primary'] + """;
        color: """ + THEME['text'] + """;
    }
    button[kind="secondary"] {
        background-color: """ + THEME['grid'] + """;
        color: """ + THEME['text'] + """;
    }
    .stDownloadButton > button {
        background-color: """ + THEME['primary'] + """ !important;
        color: """ + THEME['text'] + """ !important;
    }
    .streamlit-expanderHeader {
        color: """ + THEME['text'] + """ !important;
        background-color: """ + THEME['grid'] + """ !important;
    }
    .stDataFrame {
        color: """ + THEME['text'] + """;
    }
    .stDataFrame [data-testid="stTable"] {
        background-color: """ + THEME['grid'] + """;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: transparent !important;
        color: black !important;
    }
    
    .stRadio > div > div > div {
        background-color: transparent !important;
    }
    
    /* Header Alignment */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        text-align: center !important;
    }
    
    </style>
    """

def get_color_sequence():
    """Get color sequence for consistent chart colors."""
    return [
        THEME['primary'],
        THEME['secondary'],
        THEME['accent'],
        THEME['warning'],
        THEME['success'],
        THEME['danger']
    ]
