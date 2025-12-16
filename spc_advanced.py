import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import os

# ============================================
# AYARLAR
# ============================================
KLASOR_YOLU = r"C:\Users\ENGINME1\Desktop\spc_2"
SABIT_DOSYA_ADI = os.path.join(KLASOR_YOLU, "SPC 26.11.2025.xlsx")
LOGO_DOSYA_ADI = os.path.join(KLASOR_YOLU, "logo.png")
SIRKET_ISMI = "MALHOTRA KABLO"

# Sayfa ayarları
st.set_page_config(
    page_title=f"SPC - {SIRKET_ISMI}", 
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ============================================
# RENK PALETİ (Renk körü dostu)
# ============================================
COLORS = {
    'primary': '#2563eb',      # Mavi
    'success': '#16a34a',      # Yeşil
    'warning': '#d97706',      # Turuncu
    'danger': '#dc2626',       # Kırmızı
    'purple': '#7c3aed',       # Mor
    'gray': '#6b7280',         # Gri
    'light_blue': '#93c5fd',
    'light_green': '#86efac',
    'light_red': '#fca5a5',
    'background': '#f8fafc'
}

# ============================================
# SÜTUN EŞLEŞTİRME
# ============================================
COL_DATE = 'TARİH'
COL_GROUP = 'KESİT'
COL_MACHINE = 'MAKİNE'

PARAM_MAP = {
    'Birim Ağırlık (g/m)': {
        'sutun': 'TARTILAN BİRİM AĞIRLIK',
        'aciklama': 'Kablonun metre başına ağırlığı. Üretim kalitesinin temel göstergesidir.',
        'birim': 'g/m',
        'icon': '⚖️'
    },
    'Başlangıç Birim Ağırlık (g/m)': {
        'sutun': 'BAŞLANGIÇ BİRİM AĞIRLIK',
        'aciklama': 'Üretim başlangıcında ölçülen birim ağırlık değeri.',
        'birim': 'g/m',
        'icon': '⚖️'
    },
    'Başlangıç Direnç (Ω)': {
        'sutun': 'KALİTE BAŞLANGIÇ ÖLÇÜLEN DİRENÇ',
        'aciklama': 'Üretim başlangıcında ölçülen elektriksel direnç değeri.',
        'birim': 'Ω (Ohm)',
        'icon': '🔌'
    },
    'Bitiş Direnç (Ω)': {
        'sutun': 'KALİTE BİTİŞ ÖLÇÜLEN DİRENÇ',
        'aciklama': 'Üretim bitişinde ölçülen elektriksel direnç değeri.',
        'birim': 'Ω (Ohm)',
        'icon': '🔌'
    },
    'Direnç Farkı (Başlangıç-Bitiş)': {
        'sutun': 'DIRENC_FARKI',
        'aciklama': 'Başlangıç ve bitiş direnci arasındaki fark.',
        'birim': 'Ω (Ohm)',
        'icon': '📐'
    },
    'CR Farkı (Başlangıç-Bitiş)': {
        'sutun': 'BAŞLANGIÇ - BİTİŞ CR',
        'aciklama': 'Başlangıç ve bitiş CR değerleri arasındaki fark.',
        'birim': '-',
        'icon': '📊'
    },
    'Başlangıç CR': {
        'sutun': 'BAŞLANGIÇ CR',
        'aciklama': 'Üretim başlangıcındaki CR değeri.',
        'birim': '-',
        'icon': '📈'
    },
    'Bitiş CR': {
        'sutun': 'BİTİŞ CR',
        'aciklama': 'Üretim bitişindeki CR değeri.',
        'birim': '-',
        'icon': '📉'
    }
}

# ============================================
# CSS STİLLERİ
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #2563eb;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e3a5f;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem 0;
        border-left: 4px solid #2563eb;
        padding-left: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin: 0.5rem 0;
        border-left: 4px solid #2563eb;
    }
    
    .metric-card.excellent { border-left-color: #16a34a; background: linear-gradient(to right, #f0fdf4, white); }
    .metric-card.good { border-left-color: #22c55e; }
    .metric-card.warning { border-left-color: #d97706; background: linear-gradient(to right, #fffbeb, white); }
    .metric-card.danger { border-left-color: #dc2626; background: linear-gradient(to right, #fef2f2, white); }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1e3a5f;
        margin: 0.3rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .metric-desc {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }
    
    .info-box {
        background: #f1f5f9;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
        margin: 0.75rem 0;
        font-size: 0.9rem;
    }
    
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    
    .alert-success { background: #f0fdf4; border: 1px solid #86efac; color: #166534; }
    .alert-warning { background: #fffbeb; border: 1px solid #fcd34d; color: #92400e; }
    .alert-danger { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; }
    
    .stat-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    
    .stat-table th {
        background: #2563eb;
        color: white;
        padding: 10px 12px;
        text-align: left;
        font-weight: 500;
    }
    
    .stat-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .stat-table tr:hover { background: #f8fafc; }
    
    .gauge-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .rule-violation {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
        color: #991b1b;
    }
    
    .rule-ok {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# GELİŞMİŞ YARDIMCI FONKSİYONLAR
# ============================================

def calculate_spc_metrics(data, usl=None, lsl=None):
    """Tüm SPC metriklerini hesapla - GELİŞTİRİLMİŞ"""
    n = len(data)
    if n < 2:
        return None
    
    mean = data.mean()
    std_sample = data.std(ddof=1)
    
    # Moving Range ile sigma tahmini (within sigma)
    mr = data.diff().abs().dropna()
    mr_mean = mr.mean() if len(mr) > 0 else 0
    sigma_within = mr_mean / 1.128 if mr_mean > 0 else std_sample
    
    # Kontrol limitleri
    ucl = mean + 3 * sigma_within
    lcl = mean - 3 * sigma_within
    mr_ucl = 3.267 * mr_mean if mr_mean > 0 else 0
    
    # Sigma bantları
    sigma_bands = {
        '1sigma_upper': mean + sigma_within,
        '1sigma_lower': mean - sigma_within,
        '2sigma_upper': mean + 2 * sigma_within,
        '2sigma_lower': mean - 2 * sigma_within,
    }
    
    # Yeterlilik indeksleri
    cp, cpk, cpu, cpl = None, None, None, None
    pp, ppk, ppu, ppl = None, None, None, None
    ppm, ppm_upper, ppm_lower = None, None, None
    sigma_level = None
    
    if usl is not None and lsl is not None and sigma_within > 0:
        # Cp ve Cpk (within sigma ile - kısa vadeli)
        cp = (usl - lsl) / (6 * sigma_within)
        cpu = (usl - mean) / (3 * sigma_within)
        cpl = (mean - lsl) / (3 * sigma_within)
        cpk = min(cpu, cpl)
        
        # Pp ve Ppk (overall sigma ile - uzun vadeli)
        if std_sample > 0:
            pp = (usl - lsl) / (6 * std_sample)
            ppu = (usl - mean) / (3 * std_sample)
            ppl = (mean - lsl) / (3 * std_sample)
            ppk = min(ppu, ppl)
        
        # PPM hesaplama
        z_upper = (usl - mean) / sigma_within
        z_lower = (mean - lsl) / sigma_within
        ppm_upper = (1 - stats.norm.cdf(z_upper)) * 1e6
        ppm_lower = stats.norm.cdf(-z_lower) * 1e6
        ppm = ppm_upper + ppm_lower
        
        # Sigma seviyesi (Cpk'dan)
        if cpk is not None and cpk > 0:
            sigma_level = cpk * 3
    
    # Normallik testi (Shapiro-Wilk)
    normality_stat, normality_p = None, None
    if 3 <= n <= 5000:
        try:
            normality_stat, normality_p = stats.shapiro(data)
        except:
            pass
    
    # Çarpıklık ve basıklık
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    return {
        'mean': mean,
        'std': std_sample,
        'sigma_within': sigma_within,
        'ucl': ucl,
        'lcl': lcl,
        'mr_mean': mr_mean,
        'mr_ucl': mr_ucl,
        'sigma_bands': sigma_bands,
        'cp': cp,
        'cpk': cpk,
        'cpu': cpu,
        'cpl': cpl,
        'pp': pp,
        'ppk': ppk,
        'ppu': ppu,
        'ppl': ppl,
        'ppm': ppm,
        'ppm_upper': ppm_upper,
        'ppm_lower': ppm_lower,
        'sigma_level': sigma_level,
        'normality_stat': normality_stat,
        'normality_p': normality_p,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'n': n,
        'min': data.min(),
        'max': data.max(),
        'median': data.median(),
        'q1': data.quantile(0.25),
        'q3': data.quantile(0.75),
        'iqr': data.quantile(0.75) - data.quantile(0.25)
    }

def check_western_electric_rules(data, mean, sigma):
    """Western Electric kurallarını kontrol et"""
    violations = []
    n = len(data)
    
    if n < 8:
        return violations, []
    
    # Kural 1: Tek nokta 3σ dışında
    rule1_violations = []
    for i, val in enumerate(data):
        if abs(val - mean) > 3 * sigma:
            rule1_violations.append(i)
    if rule1_violations:
        violations.append({
            'rule': 1,
            'desc': '3σ dışında nokta',
            'points': rule1_violations,
            'severity': 'high'
        })
    
    # Kural 2: 3 ardışık noktadan 2'si 2σ-3σ arasında (aynı tarafta)
    rule2_violations = []
    for i in range(2, n):
        window = data.iloc[i-2:i+1]
        above_2sigma = ((window - mean) > 2 * sigma).sum()
        below_2sigma = ((mean - window) > 2 * sigma).sum()
        if above_2sigma >= 2 or below_2sigma >= 2:
            rule2_violations.append(i)
    if rule2_violations:
        violations.append({
            'rule': 2,
            'desc': '3 noktadan 2si 2σ-3σ bölgesinde',
            'points': rule2_violations,
            'severity': 'medium'
        })
    
    # Kural 3: 5 ardışık noktadan 4'ü 1σ-2σ arasında (aynı tarafta)
    rule3_violations = []
    for i in range(4, n):
        window = data.iloc[i-4:i+1]
        above_1sigma = ((window - mean) > sigma).sum()
        below_1sigma = ((mean - window) > sigma).sum()
        if above_1sigma >= 4 or below_1sigma >= 4:
            rule3_violations.append(i)
    if rule3_violations:
        violations.append({
            'rule': 3,
            'desc': '5 noktadan 4ü 1σ-2σ bölgesinde',
            'points': rule3_violations,
            'severity': 'medium'
        })
    
    # Kural 4: 8 ardışık nokta ortalamanın aynı tarafında
    rule4_violations = []
    for i in range(7, n):
        window = data.iloc[i-7:i+1]
        if all(window > mean) or all(window < mean):
            rule4_violations.append(i)
    if rule4_violations:
        violations.append({
            'rule': 4,
            'desc': '8 ardışık nokta aynı tarafta',
            'points': rule4_violations,
            'severity': 'medium'
        })
    
    # Kural 5: 6 ardışık nokta artan veya azalan trend
    rule5_violations = []
    for i in range(5, n):
        window = data.iloc[i-5:i+1].values
        diffs = np.diff(window)
        if all(diffs > 0) or all(diffs < 0):
            rule5_violations.append(i)
    if rule5_violations:
        violations.append({
            'rule': 5,
            'desc': '6 ardışık nokta trend oluşturmuş',
            'points': rule5_violations,
            'severity': 'medium'
        })
    
    # Tüm ihlal noktalarını topla
    all_violation_points = set()
    for v in violations:
        all_violation_points.update(v['points'])
    
    return violations, list(all_violation_points)

def get_cpk_info(cpk):
    """Cpk değerine göre detaylı bilgi"""
    if cpk is None:
        return {
            'color': COLORS['gray'],
            'status': 'Belirsiz',
            'icon': '❓',
            'class': '',
            'desc': 'Tolerans limitleri girilmedi',
            'action': 'USL ve LSL değerlerini girin'
        }
    elif cpk >= 1.67:
        return {
            'color': COLORS['success'],
            'status': 'Mükemmel',
            'icon': '🌟',
            'class': 'excellent',
            'desc': '6σ seviyesine yakın, dünya standartlarında',
            'action': 'Mükemmelliği sürdürün'
        }
    elif cpk >= 1.33:
        return {
            'color': '#22c55e',
            'status': 'İyi',
            'icon': '✅',
            'class': 'good',
            'desc': 'Süreç yeterli, hedeflere uygun',
            'action': 'İzlemeye devam edin'
        }
    elif cpk >= 1.00:
        return {
            'color': COLORS['warning'],
            'status': 'Kabul Edilebilir',
            'icon': '⚠️',
            'class': 'warning',
            'desc': 'Süreç sınırda, iyileştirme gerekli',
            'action': 'İyileştirme planı hazırlayın'
        }
    else:
        return {
            'color': COLORS['danger'],
            'status': 'Yetersiz',
            'icon': '❌',
            'class': 'danger',
            'desc': 'Süreç yetersiz, acil müdahale gerekli',
            'action': 'Acil düzeltici faaliyet başlatın'
        }

def create_gauge_chart(value, title, min_val=0, max_val=2.5, thresholds=[1.0, 1.33, 1.67]):
    """Cpk için gauge chart oluştur"""
    if value is None:
        value = 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={'font': {'size': 36, 'color': '#1e3a5f'}, 'suffix': '', 'valueformat': '.2f'},
        title={'text': title, 'font': {'size': 16, 'color': '#64748b'}},
        delta={'reference': 1.33, 'increasing': {'color': COLORS['success']}, 'decreasing': {'color': COLORS['danger']}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 2, 'tickcolor': "#64748b",
                     'tickvals': [0, 0.5, 1.0, 1.33, 1.67, 2.0, 2.5],
                     'ticktext': ['0', '0.5', '1.0', '1.33', '1.67', '2.0', '2.5']},
            'bar': {'color': get_cpk_info(value)['color'], 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 1.0], 'color': '#fecaca'},
                {'range': [1.0, 1.33], 'color': '#fed7aa'},
                {'range': [1.33, 1.67], 'color': '#bbf7d0'},
                {'range': [1.67, 2.5], 'color': '#86efac'}
            ],
            'threshold': {
                'line': {'color': "#1e3a5f", 'width': 3},
                'thickness': 0.8,
                'value': 1.33
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial'}
    )
    
    return fig

def create_capability_histogram(data, mean, sigma, usl=None, lsl=None, title="Yetenek Histogramı"):
    """Gelişmiş capability histogram"""
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=35,
        name='Dağılım',
        marker_color=COLORS['primary'],
        opacity=0.7,
        hovertemplate='<b>Aralık:</b> %{x}<br><b>Frekans:</b> %{y}<extra></extra>'
    ))
    
    # Normal eğri
    x_range = np.linspace(data.min() - sigma, data.max() + sigma, 200)
    y_norm = stats.norm.pdf(x_range, mean, sigma)
    scale = len(data) * (data.max() - data.min()) / 35
    
    fig.add_trace(go.Scatter(
        x=x_range, y=y_norm * scale,
        mode='lines',
        name='Normal Dağılım',
        line=dict(color=COLORS['danger'], width=3)
    ))
    
    # Ortalama çizgisi
    fig.add_vline(x=mean, line_color=COLORS['success'], line_width=3,
                  annotation_text=f"X̄ = {mean:.4f}", 
                  annotation_position="top",
                  annotation_font_size=12,
                  annotation_font_color=COLORS['success'])
    
    # Spec limitleri
    if usl is not None:
        fig.add_vline(x=usl, line_color=COLORS['warning'], line_width=3, line_dash='dash',
                      annotation_text=f"USL = {usl:.4f}",
                      annotation_position="top right",
                      annotation_font_size=11)
        # USL dışı alan
        fig.add_vrect(x0=usl, x1=data.max() + sigma, fillcolor="rgba(220, 38, 38, 0.15)", line_width=0)
    
    if lsl is not None:
        fig.add_vline(x=lsl, line_color=COLORS['warning'], line_width=3, line_dash='dash',
                      annotation_text=f"LSL = {lsl:.4f}",
                      annotation_position="top left",
                      annotation_font_size=11)
        # LSL dışı alan
        fig.add_vrect(x0=data.min() - sigma, x1=lsl, fillcolor="rgba(220, 38, 38, 0.15)", line_width=0)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#1e3a5f')),
        height=400,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=40, t=80, b=60),
        xaxis_title="Değer",
        yaxis_title="Frekans",
        font=dict(size=12)
    )
    
    return fig

def create_control_chart(data, x_axis, metrics, usl=None, lsl=None, title="I-MR Kontrol Grafiği", 
                         violation_points=None, param_name="", unit=""):
    """Gelişmiş kontrol grafiği"""
    fig = go.Figure()
    
    mean = metrics['mean']
    sigma = metrics['sigma_within']
    ucl = metrics['ucl']
    lcl = metrics['lcl']
    
    # Sigma bantları (arka plan)
    fig.add_hrect(y0=mean - sigma, y1=mean + sigma,
                  fillcolor="rgba(34, 197, 94, 0.12)", line_width=0,
                  annotation_text="±1σ (68%)", annotation_position="right",
                  annotation_font_size=10, annotation_font_color='#64748b')
    
    fig.add_hrect(y0=mean - 2*sigma, y1=mean - sigma,
                  fillcolor="rgba(234, 179, 8, 0.08)", line_width=0)
    fig.add_hrect(y0=mean + sigma, y1=mean + 2*sigma,
                  fillcolor="rgba(234, 179, 8, 0.08)", line_width=0,
                  annotation_text="±2σ (95%)", annotation_position="right",
                  annotation_font_size=10, annotation_font_color='#64748b')
    
    fig.add_hrect(y0=mean - 3*sigma, y1=mean - 2*sigma,
                  fillcolor="rgba(239, 68, 68, 0.06)", line_width=0)
    fig.add_hrect(y0=mean + 2*sigma, y1=mean + 3*sigma,
                  fillcolor="rgba(239, 68, 68, 0.06)", line_width=0,
                  annotation_text="±3σ (99.7%)", annotation_position="right",
                  annotation_font_size=10, annotation_font_color='#64748b')
    
    # Ana veri çizgisi
    fig.add_trace(go.Scatter(
        x=x_axis, y=data,
        mode='lines+markers',
        name='Ölçüm',
        line=dict(color=COLORS['primary'], width=1.5),
        marker=dict(size=6, color=COLORS['primary']),
        hovertemplate='<b>Değer:</b> %{y:.4f}<br><b>Tarih:</b> %{x}<extra></extra>'
    ))
    
    # Ortalama çizgisi
    fig.add_hline(y=mean, line_color=COLORS['success'], line_width=3,
                  annotation_text=f"X̄ = {mean:.4f}",
                  annotation_position="left",
                  annotation_font_size=13,
                  annotation_font_color=COLORS['success'])
    
    # Kontrol limitleri
    fig.add_hline(y=ucl, line_dash='dash', line_color=COLORS['danger'], line_width=2.5,
                  annotation_text=f"UCL = {ucl:.4f}",
                  annotation_position="right",
                  annotation_font_size=12,
                  annotation_font_color=COLORS['danger'])
    
    fig.add_hline(y=lcl, line_dash='dash', line_color=COLORS['danger'], line_width=2.5,
                  annotation_text=f"LCL = {lcl:.4f}",
                  annotation_position="right",
                  annotation_font_size=12,
                  annotation_font_color=COLORS['danger'])
    
    # Tolerans limitleri
    if usl is not None:
        fig.add_hline(y=usl, line_color=COLORS['warning'], line_width=3, line_dash='dot',
                      annotation_text=f"USL = {usl}",
                      annotation_position="right",
                      annotation_font_size=12,
                      annotation_font_color=COLORS['warning'])
    
    if lsl is not None:
        fig.add_hline(y=lsl, line_color=COLORS['warning'], line_width=3, line_dash='dot',
                      annotation_text=f"LSL = {lsl}",
                      annotation_position="right",
                      annotation_font_size=12,
                      annotation_font_color=COLORS['warning'])
    
    # Kontrol dışı noktalar
    out_mask = (data > ucl) | (data < lcl)
    out_points = data[out_mask]
    if len(out_points) > 0:
        out_x = x_axis[out_mask] if hasattr(x_axis, '__getitem__') else [x_axis[i] for i in out_points.index]
        fig.add_trace(go.Scatter(
            x=out_x, y=out_points,
            mode='markers',
            name='Kontrol Dışı',
            marker=dict(color=COLORS['danger'], size=14, symbol='x', 
                       line=dict(width=2, color='white')),
            hovertemplate='<b>⚠️ Kontrol Dışı</b><br>Değer: %{y:.4f}<extra></extra>'
        ))
    
    # Kural ihlali noktaları
    if violation_points and len(violation_points) > 0:
        viol_data = data.iloc[violation_points]
        if hasattr(x_axis, 'iloc'):
            viol_x = x_axis.iloc[violation_points]
        else:
            viol_x = [x_axis[i] for i in violation_points]
        
        fig.add_trace(go.Scatter(
            x=viol_x, y=viol_data,
            mode='markers',
            name='Kural İhlali',
            marker=dict(color=COLORS['purple'], size=12, symbol='diamond',
                       line=dict(width=2, color='white')),
            hovertemplate='<b>⚠️ Kural İhlali</b><br>Değer: %{y:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1e3a5f'), x=0.5),
        height=550,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                   font=dict(size=11)),
        margin=dict(l=70, r=150, t=80, b=70),
        xaxis_title="Tarih / Sıra",
        yaxis_title=f"{param_name} ({unit})" if unit else param_name,
        font=dict(size=12, family='Arial')
    )
    
    return fig

def create_mr_chart(data, x_axis, mr_mean, mr_ucl, title="Moving Range (MR) Grafiği"):
    """Moving Range grafiği"""
    mr = data.diff().abs()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_axis, y=mr,
        mode='lines+markers',
        name='Moving Range',
        line=dict(color=COLORS['purple'], width=1.5),
        marker=dict(size=5, color=COLORS['purple']),
        hovertemplate='<b>MR:</b> %{y:.4f}<extra></extra>'
    ))
    
    fig.add_hline(y=mr_mean, line_color=COLORS['success'], line_width=2.5,
                  annotation_text=f"MR̄ = {mr_mean:.4f}",
                  annotation_position="left",
                  annotation_font_size=12)
    
    fig.add_hline(y=mr_ucl, line_dash='dash', line_color=COLORS['danger'], line_width=2,
                  annotation_text=f"UCL = {mr_ucl:.4f}",
                  annotation_position="right",
                  annotation_font_size=11)
    
    # MR kontrol dışı
    mr_out = mr[mr > mr_ucl]
    if len(mr_out) > 0:
        if hasattr(x_axis, 'iloc'):
            mr_out_x = x_axis.iloc[mr_out.index]
        else:
            mr_out_x = [x_axis[i] for i in mr_out.index]
        fig.add_trace(go.Scatter(
            x=mr_out_x, y=mr_out,
            mode='markers',
            name='MR Kontrol Dışı',
            marker=dict(color=COLORS['danger'], size=12, symbol='x')
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#1e3a5f')),
        height=350,
        template="plotly_white",
        margin=dict(l=60, r=100, t=60, b=50),
        xaxis_title="Tarih / Sıra",
        yaxis_title="Moving Range",
        font=dict(size=11)
    )
    
    return fig

def count_out_of_limits(data, upper=None, lower=None):
    """Limit dışı nokta sayısı"""
    count = 0
    if upper is not None:
        count += (data > upper).sum()
    if lower is not None:
        count += (data < lower).sum()
    return int(count)

def format_number(val, decimals=4):
    """Sayı formatlama"""
    if val is None:
        return "-"
    if abs(val) >= 10000:
        return f"{val:,.1f}"
    elif abs(val) >= 1000:
        return f"{val:,.2f}"
    elif abs(val) >= 1:
        return f"{val:.{decimals}f}"
    else:
        return f"{val:.{max(decimals, 6)}f}"

# ============================================
# VERİ YÜKLEME
# ============================================
@st.cache_data
def load_data(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        return df, None
    except Exception as e:
        return None, str(e)

# ============================================
# SIDEBAR
# ============================================
if os.path.exists(LOGO_DOSYA_ADI):
    st.sidebar.image(LOGO_DOSYA_ADI, width=180)
    
st.sidebar.markdown(f"## 📊 {SIRKET_ISMI}")
st.sidebar.markdown("**Gelişmiş SPC Analiz Sistemi v2.0**")
st.sidebar.markdown("---")

# Veri yükleme
df = None
if os.path.exists(SABIT_DOSYA_ADI):
    df, error = load_data(SABIT_DOSYA_ADI)
    if error:
        st.sidebar.error(f"Hata: {error}")
else:
    uploaded = st.sidebar.file_uploader("📁 Veri Yükle", type=['xlsx', 'csv'])
    if uploaded:
        if uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

if df is None:
    st.markdown('<p class="main-header">📊 Gelişmiş SPC Analiz Sistemi</p>', unsafe_allow_html=True)
    st.info(f"📁 Lütfen veri dosyası yükleyin veya **{os.path.basename(SABIT_DOSYA_ADI)}** dosyasını klasöre ekleyin.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 SPC Nedir?
        
        **İstatistiksel Süreç Kontrolü (SPC)**, üretim süreçlerini izlemek ve kontrol altında 
        tutmak için kullanılan güçlü bir kalite yönetim aracıdır.
        
        **Bu sistem ile:**
        - Süreç değişkenliğini ölçün ve izleyin
        - Problemleri erken tespit edin
        - Kaliteyi sürekli iyileştirin
        - Müşteri memnuniyetini artırın
        """)
    
    with col2:
        st.markdown("""
        ### ⭐ Yeni Özellikler (v2.0)
        
        - 📊 **Gelişmiş Grafikler**: Daha okunaklı, interaktif
        - 🎯 **Gauge Chart**: Cpk görsel göstergesi
        - 📏 **Western Electric Kuralları**: Otomatik kural kontrolü
        - 📈 **Normallik Testi**: Shapiro-Wilk testi
        - 🔍 **Sigma Bantları**: 1σ, 2σ, 3σ görselleştirme
        - 📋 **Detaylı Raporlama**: Aksiyon önerileri
        """)
    
    st.stop()

# ============================================
# VERİ HAZIRLAMA
# ============================================
if COL_DATE in df.columns:
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors='coerce')

if 'KALİTE BAŞLANGIÇ ÖLÇÜLEN DİRENÇ' in df.columns and 'KALİTE BİTİŞ ÖLÇÜLEN DİRENÇ' in df.columns:
    df['DIRENC_FARKI'] = df['KALİTE BAŞLANGIÇ ÖLÇÜLEN DİRENÇ'] - df['KALİTE BİTİŞ ÖLÇÜLEN DİRENÇ']

df_original = df.copy()

# ============================================
# FİLTRELER
# ============================================
st.sidebar.markdown("### 🔍 Filtreler")

# Tarih filtresi
if COL_DATE in df.columns and df[COL_DATE].notna().any():
    min_date = df[COL_DATE].min().date()
    max_date = df[COL_DATE].max().date()
    
    date_range = st.sidebar.date_input(
        "📅 Tarih Aralığı",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        df = df[(df[COL_DATE].dt.date >= date_range[0]) & (df[COL_DATE].dt.date <= date_range[1])]

# Kesit filtresi
secili_kesit = 'Tümü'
if COL_GROUP in df.columns:
    kesitler = ['Tümü'] + sorted(df[COL_GROUP].dropna().unique().tolist())
    secili_kesit = st.sidebar.selectbox("📦 Kesit", kesitler)
    
    if secili_kesit != 'Tümü':
        df = df[df[COL_GROUP] == secili_kesit]

# Makine filtresi
if COL_MACHINE in df.columns:
    makineler = ['Tümü'] + sorted(df[COL_MACHINE].dropna().unique().astype(str).tolist())
    secili_makine = st.sidebar.selectbox("🏭 Makine", makineler)
    
    if secili_makine != 'Tümü':
        df = df[df[COL_MACHINE].astype(str) == secili_makine]

st.sidebar.markdown("---")
st.sidebar.metric("📊 Filtrelenmiş Kayıt", f"{len(df):,}")

# ============================================
# ANA BAŞLIK
# ============================================
st.markdown(f'<p class="main-header">📊 SPC Analiz - {SIRKET_ISMI}</p>', unsafe_allow_html=True)

# Özet kartları
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📊 Toplam Kayıt</div>
        <div class="metric-value">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    kesit_sayi = df[COL_GROUP].nunique() if COL_GROUP in df.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📦 Kesit Çeşidi</div>
        <div class="metric-value">{kesit_sayi}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    makine_sayi = df[COL_MACHINE].nunique() if COL_MACHINE in df.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🏭 Makine Sayısı</div>
        <div class="metric-value">{makine_sayi}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    if COL_DATE in df.columns and len(df) > 0 and df[COL_DATE].notna().any():
        gun = (df[COL_DATE].max() - df[COL_DATE].min()).days + 1
    else:
        gun = 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📅 Analiz Süresi</div>
        <div class="metric-value">{gun} gün</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PARAMETRE SEÇİMİ
# ============================================
st.markdown('<p class="section-header">🔬 Parametre Seçimi</p>', unsafe_allow_html=True)

mevcut_params = {}
for key, val in PARAM_MAP.items():
    if val['sutun'] in df.columns:
        mevcut_params[key] = val

if not mevcut_params:
    st.error("❌ Analiz edilebilecek parametre bulunamadı!")
    st.stop()

col_select, col_info = st.columns([2, 3])

with col_select:
    secili_param = st.selectbox(
        "📈 Analiz edilecek parametre:",
        list(mevcut_params.keys())
    )

param_info = mevcut_params[secili_param]

with col_info:
    st.markdown(f"""
    <div class="info-box">
        <b>{param_info['icon']} {secili_param}</b><br>
        {param_info['aciklama']}<br>
        <small><b>Birim:</b> {param_info['birim']}</small>
    </div>
    """, unsafe_allow_html=True)

# Tolerans ayarları
col_tol1, col_tol2, col_tol3 = st.columns([1, 1, 1])

with col_tol1:
    use_spec = st.checkbox("📏 Tolerans Limitleri Kullan", value=False)

usl, lsl = None, None
if use_spec:
    with col_tol2:
        usl = st.number_input("USL (Üst Tolerans)", value=None, format="%.4f")
    with col_tol3:
        lsl = st.number_input("LSL (Alt Tolerans)", value=None, format="%.4f")

# ============================================
# VERİ HAZIRLAMA VE HESAPLAMA
# ============================================
secili_sutun = param_info['sutun']

if COL_DATE in df.columns:
    df = df.sort_values(COL_DATE)

data = df[secili_sutun].dropna()

if len(data) < 2:
    st.warning("⚠️ Yeterli veri yok (en az 2 ölçüm gerekli)")
    st.stop()

# SPC hesaplamaları
metrics = calculate_spc_metrics(data, usl, lsl)

if metrics is None:
    st.error("❌ Hesaplama hatası!")
    st.stop()

# Western Electric kuralları kontrolü
violations, violation_points = check_western_electric_rules(
    data, metrics['mean'], metrics['sigma_within']
)

# ============================================
# ANALİZ SONUÇLARI
# ============================================
st.markdown('<p class="section-header">📊 Analiz Sonuçları</p>', unsafe_allow_html=True)

# İstatistikler
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📈 Ortalama (X̄)</div>
        <div class="metric-value">{format_number(metrics['mean'])}</div>
        <div class="metric-desc">Merkezi değer</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📊 Std Sapma (σ)</div>
        <div class="metric-value">{format_number(metrics['sigma_within'])}</div>
        <div class="metric-desc">Within sigma</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⬇️ Minimum</div>
        <div class="metric-value">{format_number(metrics['min'])}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⬆️ Maksimum</div>
        <div class="metric-value">{format_number(metrics['max'])}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    out_ctrl = count_out_of_limits(data, metrics['ucl'], metrics['lcl'])
    card_class = "danger" if out_ctrl > 0 else "excellent"
    st.markdown(f"""
    <div class="metric-card {card_class}">
        <div class="metric-label">🔴 Kontrol Dışı</div>
        <div class="metric-value">{out_ctrl}</div>
        <div class="metric-desc">{out_ctrl/len(data)*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# Kontrol limitleri
col_ucl, col_lcl, col_range, col_n = st.columns(4)

with col_ucl:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {COLORS['danger']};">
        <div class="metric-label">🔺 UCL</div>
        <div class="metric-value" style="color: {COLORS['danger']};">{format_number(metrics['ucl'])}</div>
        <div class="metric-desc">X̄ + 3σ</div>
    </div>
    """, unsafe_allow_html=True)

with col_lcl:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {COLORS['danger']};">
        <div class="metric-label">🔻 LCL</div>
        <div class="metric-value" style="color: {COLORS['danger']};">{format_number(metrics['lcl'])}</div>
        <div class="metric-desc">X̄ - 3σ</div>
    </div>
    """, unsafe_allow_html=True)

with col_range:
    range_val = metrics['max'] - metrics['min']
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">↔️ Aralık</div>
        <div class="metric-value">{format_number(range_val)}</div>
        <div class="metric-desc">Max - Min</div>
    </div>
    """, unsafe_allow_html=True)

with col_n:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📋 Veri Sayısı</div>
        <div class="metric-value">{metrics['n']:,}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# SÜREÇ YETERLİLİĞİ (Cpk)
# ============================================
if metrics['cpk'] is not None:
    st.markdown('<p class="section-header">⭐ Süreç Yeterliliği</p>', unsafe_allow_html=True)
    
    cpk_info = get_cpk_info(metrics['cpk'])
    
    col_gauge, col_metrics, col_info = st.columns([1.2, 1.5, 1.3])
    
    with col_gauge:
        gauge_fig = create_gauge_chart(metrics['cpk'], "Cpk Değeri")
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col_metrics:
        st.markdown("#### 📊 Yeterlilik İndeksleri")
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cp (Potansiyel)</div>
                <div class="metric-value">{format_number(metrics['cp'], 2)}</div>
                <div class="metric-desc">Merkezleme hariç</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Pp (Uzun Vadeli)</div>
                <div class="metric-value">{format_number(metrics['pp'], 2)}</div>
                <div class="metric-desc">Overall sigma ile</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m2:
            st.markdown(f"""
            <div class="metric-card {cpk_info['class']}">
                <div class="metric-label">Cpk {cpk_info['icon']}</div>
                <div class="metric-value" style="color: {cpk_info['color']};">{format_number(metrics['cpk'], 2)}</div>
                <div class="metric-desc">{cpk_info['status']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Ppk</div>
                <div class="metric-value">{format_number(metrics['ppk'], 2)}</div>
                <div class="metric-desc">Uzun vadeli yetenek</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_info:
        st.markdown("#### 📋 Değerlendirme")
        
        # Durum kutusu
        alert_class = f"alert-{'success' if cpk_info['class'] in ['excellent', 'good'] else 'warning' if cpk_info['class'] == 'warning' else 'danger'}"
        st.markdown(f"""
        <div class="alert-box {alert_class}">
            <b>{cpk_info['icon']} {cpk_info['status']}</b><br>
            {cpk_info['desc']}<br><br>
            <b>Önerilen Aksiyon:</b><br>
            {cpk_info['action']}
        </div>
        """, unsafe_allow_html=True)
        
        # PPM ve Sigma
        if metrics['ppm'] is not None:
            ppm_text = f"{metrics['ppm']:,.0f}" if metrics['ppm'] >= 1 else "< 1"
            sigma_text = f"{metrics['sigma_level']:.1f}σ" if metrics['sigma_level'] else "-"
            st.markdown(f"""
            <div class="info-box">
                <b>📊 PPM (Milyonda Hata):</b> {ppm_text}<br>
                <b>📈 Sigma Seviyesi:</b> {sigma_text}
            </div>
            """, unsafe_allow_html=True)
        
        # Tolerans dışı
        if usl and lsl:
            out_spec = count_out_of_limits(data, usl, lsl)
            if out_spec > 0:
                st.markdown(f"""
                <div class="alert-box alert-danger">
                    <b>⚠️ Tolerans Dışı:</b> {out_spec} adet ({out_spec/len(data)*100:.1f}%)
                </div>
                """, unsafe_allow_html=True)

# ============================================
# NORMALLİK TESTİ
# ============================================
if metrics['normality_p'] is not None:
    with st.expander("📈 Normallik Testi (Shapiro-Wilk)", expanded=False):
        col_n1, col_n2 = st.columns(2)
        
        with col_n1:
            is_normal = metrics['normality_p'] > 0.05
            st.markdown(f"""
            <div class="metric-card {'excellent' if is_normal else 'warning'}">
                <div class="metric-label">p-değeri</div>
                <div class="metric-value">{metrics['normality_p']:.4f}</div>
                <div class="metric-desc">{'✅ Normal dağılım (p > 0.05)' if is_normal else '⚠️ Normal değil (p ≤ 0.05)'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_n2:
            st.markdown(f"""
            **Çarpıklık (Skewness):** {metrics['skewness']:.3f}  
            {'(Sağa çarpık)' if metrics['skewness'] > 0.5 else '(Sola çarpık)' if metrics['skewness'] < -0.5 else '(Simetrik)'}
            
            **Basıklık (Kurtosis):** {metrics['kurtosis']:.3f}  
            {'(Sivri)' if metrics['kurtosis'] > 0.5 else '(Basık)' if metrics['kurtosis'] < -0.5 else '(Normal)'}
            """)
        
        if not is_normal:
            st.warning("""
            ⚠️ **Dikkat:** Veriler normal dağılmıyor. SPC analizi normal dağılım varsayımına dayanır.
            Cp/Cpk değerleri dikkatli yorumlanmalıdır. Box-Cox dönüşümü düşünülebilir.
            """)

# ============================================
# WESTERN ELECTRIC KURALLARI
# ============================================
if len(data) >= 8:
    with st.expander("📏 Western Electric Kuralları", expanded=len(violations) > 0):
        if not violations:
            st.markdown("""
            <div class="rule-ok">
                ✅ <b>Tüm kurallar geçti!</b> Süreç istatistiksel kontrol altında.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rule-violation">
                ⚠️ <b>{len(violations)} kural ihlali tespit edildi!</b>
            </div>
            """, unsafe_allow_html=True)
            
            for v in violations:
                severity_icon = "🔴" if v['severity'] == 'high' else "🟡"
                st.markdown(f"""
                <div class="rule-violation">
                    {severity_icon} <b>Kural {v['rule']}:</b> {v['desc']}<br>
                    <small>İhlal noktası sayısı: {len(v['points'])}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
        ---
        **Western Electric Kuralları:**
        1. Tek nokta 3σ dışında
        2. 3 ardışık noktadan 2'si 2σ-3σ arasında (aynı tarafta)
        3. 5 ardışık noktadan 4'ü 1σ-2σ arasında (aynı tarafta)
        4. 8 ardışık nokta ortalamanın aynı tarafında
        5. 6 ardışık nokta artan veya azalan trend
        """)

# ============================================
# GRAFİKLER
# ============================================
st.markdown('<p class="section-header">📈 Grafikler</p>', unsafe_allow_html=True)

# X ekseni
if COL_DATE in df.columns and df[COL_DATE].notna().any():
    x_axis = df.loc[data.index, COL_DATE]
else:
    x_axis = list(range(len(data)))

# Tab yapısı
tab1, tab2, tab3 = st.tabs(["📈 Kontrol Grafiği", "📊 Histogram", "📉 Moving Range"])

with tab1:
    control_fig = create_control_chart(
        data, x_axis, metrics, usl, lsl,
        title=f"I-MR Kontrol Grafiği - {secili_param}",
        violation_points=violation_points,
        param_name=secili_param,
        unit=param_info['birim']
    )
    st.plotly_chart(control_fig, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
        <b>📖 Grafik Okuma Rehberi:</b><br>
        • <span style="color: #16a34a;">■</span> <b>Yeşil çizgi:</b> Ortalama (X̄)<br>
        • <span style="color: #dc2626;">- -</span> <b>Kırmızı kesikli:</b> Kontrol limitleri (UCL/LCL = ±3σ)<br>
        • <span style="color: #d97706;">···</span> <b>Turuncu noktalı:</b> Tolerans limitleri (USL/LSL)<br>
        • <span style="color: #dc2626;">✕</span> <b>Kırmızı X:</b> Kontrol dışı noktalar<br>
        • <span style="color: #7c3aed;">◆</span> <b>Mor elmas:</b> Kural ihlali noktaları
    </div>
    """, unsafe_allow_html=True)

with tab2:
    hist_fig = create_capability_histogram(
        data, metrics['mean'], metrics['sigma_within'], usl, lsl,
        title=f"Yetenek Histogramı - {secili_param}"
    )
    st.plotly_chart(hist_fig, use_container_width=True)
    
    if usl and lsl:
        st.markdown(f"""
        <div class="info-box">
            <b>📊 Dağılım Özeti:</b><br>
            • Tolerans aralığı: {lsl:.4f} - {usl:.4f} (Genişlik: {usl-lsl:.4f})<br>
            • Süreç yayılımı (6σ): {6*metrics['sigma_within']:.4f}<br>
            • Merkezden sapma: {abs(metrics['mean'] - (usl+lsl)/2):.4f}
        </div>
        """, unsafe_allow_html=True)

with tab3:
    mr_fig = create_mr_chart(
        data, x_axis, metrics['mr_mean'], metrics['mr_ucl'],
        title="Moving Range (MR) Grafiği"
    )
    st.plotly_chart(mr_fig, use_container_width=True)
    
    mr_out_count = count_out_of_limits(data.diff().abs().dropna(), metrics['mr_ucl'], None)
    if mr_out_count > 0:
        st.warning(f"⚠️ {mr_out_count} adet MR değeri kontrol limitinin üzerinde. Bu, ani değişimlerin varlığını gösterir.")

# ============================================
# MAKİNE KARŞILAŞTIRMA
# ============================================
if COL_MACHINE in df.columns and df[COL_MACHINE].nunique() > 1:
    st.markdown('<p class="section-header">🏭 Makine Karşılaştırma</p>', unsafe_allow_html=True)
    
    machine_stats = df.groupby(COL_MACHINE)[secili_sutun].agg(['mean', 'std', 'count', 'min', 'max']).round(4)
    machine_stats.columns = ['Ortalama', 'Std Sapma', 'Kayıt', 'Min', 'Max']
    machine_stats = machine_stats.sort_values('Kayıt', ascending=False)
    
    col_table, col_box = st.columns([1, 2])
    
    with col_table:
        st.dataframe(machine_stats, use_container_width=True, height=400)
    
    with col_box:
        fig_box = px.box(
            df, x=COL_MACHINE, y=secili_sutun,
            color=COL_MACHINE,
            title=f"Makine Bazlı {secili_param} Dağılımı"
        )
        
        fig_box.add_hline(y=metrics['mean'], line_color=COLORS['success'], line_dash='dash',
                          annotation_text="Genel Ort.")
        if usl:
            fig_box.add_hline(y=usl, line_color=COLORS['warning'], annotation_text="USL")
        if lsl:
            fig_box.add_hline(y=lsl, line_color=COLORS['warning'], annotation_text="LSL")
        
        fig_box.update_layout(
            height=450,
            template="plotly_white",
            showlegend=False,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig_box, use_container_width=True)

# ============================================
# ÖZET RAPOR
# ============================================
with st.expander("📋 Özet Rapor", expanded=False):
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown(f"""
        ### 📊 Temel İstatistikler
        
        | Metrik | Değer |
        |--------|-------|
        | Veri Sayısı | {metrics['n']:,} |
        | Ortalama (X̄) | {format_number(metrics['mean'])} |
        | Standart Sapma (σ) | {format_number(metrics['sigma_within'])} |
        | Minimum | {format_number(metrics['min'])} |
        | Maksimum | {format_number(metrics['max'])} |
        | Medyan | {format_number(metrics['median'])} |
        | IQR | {format_number(metrics['iqr'])} |
        """)
    
    with col_r2:
        st.markdown(f"""
        ### 📏 Kontrol Limitleri
        
        | Limit | Değer |
        |-------|-------|
        | UCL (Üst Kontrol) | {format_number(metrics['ucl'])} |
        | LCL (Alt Kontrol) | {format_number(metrics['lcl'])} |
        | +2σ | {format_number(metrics['sigma_bands']['2sigma_upper'])} |
        | -2σ | {format_number(metrics['sigma_bands']['2sigma_lower'])} |
        | +1σ | {format_number(metrics['sigma_bands']['1sigma_upper'])} |
        | -1σ | {format_number(metrics['sigma_bands']['1sigma_lower'])} |
        """)
    
    if metrics['cpk'] is not None:
        st.markdown(f"""
        ### ⭐ Yeterlilik İndeksleri
        
        | İndeks | Değer | Durum |
        |--------|-------|-------|
        | Cp | {format_number(metrics['cp'], 2)} | {'✅' if metrics['cp'] and metrics['cp'] >= 1.33 else '⚠️'} |
        | Cpk | {format_number(metrics['cpk'], 2)} | {get_cpk_info(metrics['cpk'])['icon']} {get_cpk_info(metrics['cpk'])['status']} |
        | Pp | {format_number(metrics['pp'], 2)} | {'✅' if metrics['pp'] and metrics['pp'] >= 1.33 else '⚠️'} |
        | Ppk | {format_number(metrics['ppk'], 2)} | {'✅' if metrics['ppk'] and metrics['ppk'] >= 1.33 else '⚠️'} |
        | PPM | {metrics['ppm']:,.0f if metrics['ppm'] and metrics['ppm'] >= 1 else '< 1'} | - |
        | Sigma Seviyesi | {f"{metrics['sigma_level']:.1f}σ" if metrics['sigma_level'] else '-'} | - |
        """)

# Footer
st.markdown("---")
st.markdown(f"""
<center>
<small>
📊 <b>Gelişmiş SPC Analiz Sistemi v2.0</b> | {SIRKET_ISMI} | 
Analiz: <b>{len(data):,}</b> kayıt | 
{f"Kesit: <b>{secili_kesit}</b> | " if secili_kesit != 'Tümü' else ""}
Parametre: <b>{secili_param}</b>
</small>
</center>
""", unsafe_allow_html=True)
