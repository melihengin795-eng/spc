import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy import stats
import os
import glob

# ============================================
# AYARLAR
# ============================================
KLASOR_YOLU = r"C:\Users\ENGINME1\Desktop\spc_2"
SABIT_DOSYA_ADI = os.path.join(KLASOR_YOLU, "SPC 26.11.2025.xlsx")
LOGO_DOSYA_ADI = os.path.join(KLASOR_YOLU, "logo.png")
SIRKET_ISMI = "MALHOTRA KABLO"

st.set_page_config(
    page_title=f"SPC - {SIRKET_ISMI}",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ============================================
# RENK PALETİ
# ============================================
COLORS = {
    'primary': '#2563eb',
    'success': '#16a34a',
    'warning': '#d97706',
    'danger': '#dc2626',
    'purple': '#7c3aed',
    'gray': '#6b7280',
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

# ✅ PARAM_MAP sütun adlarını "temiz" tutuyoruz (newline yok, fazla boşluk yok)
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

EXCLUDED_COLUMNS = ['TARİH', 'BARKOD NUMARASI', 'MAKİNE', 'KESİT', 'ID', 'INDEX']

# ============================================
# CSS
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
# YARDIMCI - KOLON NORMALİZASYONU
# ============================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # \n -> boşluk, çoklu boşluk -> tek boşluk, trim
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df

def pick_latest_spc_file(folder: str) -> str | None:
    files = glob.glob(os.path.join(folder, "SPC*.xlsx"))
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return files[0]

# ============================================
# GELİŞMİŞ YARDIMCI FONKSİYONLAR
# ============================================
def get_available_numeric_columns(df):
    numeric_cols = []
    for col in df.columns:
        col_upper = str(col).upper().replace('\n', ' ').strip()

        # Hariç tutulacak sütunları atla
        if any(exc.upper() in col_upper for exc in EXCLUDED_COLUMNS):
            continue

        if np.issubdtype(df[col].dtype, np.number):
            numeric_cols.append(col)
        else:
            try:
                # virgüllü sayı olabilir
                s = df[col].astype(str).str.replace(",", ".", regex=False)
                numeric_vals = pd.to_numeric(s, errors='coerce')
                valid_ratio = numeric_vals.notna().sum() / max(len(df), 1)
                if valid_ratio >= 0.5:
                    numeric_cols.append(col)
            except:
                pass
    return numeric_cols

def convert_column_to_numeric(df, col):
    if np.issubdtype(df[col].dtype, np.number):
        return df[col]
    s = df[col].astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors='coerce')

def get_column_info(col_name, df):
    col_normalized = ' '.join(str(col_name).split()).upper()

    for param_name, param_info in PARAM_MAP.items():
        param_sutun_normalized = ' '.join(str(param_info['sutun']).split()).upper()
        if param_sutun_normalized == col_normalized:
            return {
                'display_name': param_name,
                'sutun': col_name,
                'aciklama': param_info['aciklama'],
                'birim': param_info['birim'],
                'icon': param_info['icon'],
                'predefined': True
            }

    col_display = ' '.join(str(col_name).split())
    birim = '-'
    icon = '📊'
    aciklama = f"'{col_display}' sütunu için SPC analizi"

    col_upper = col_display.upper()
    if 'DİRENÇ' in col_upper or 'DIRENÇ' in col_upper or 'OHM' in col_upper:
        birim = 'Ω (Ohm)'; icon = '🔌'; aciklama = 'Elektriksel direnç ölçümü'
    elif 'AĞIRLIK' in col_upper or 'WEIGHT' in col_upper:
        birim = 'g/m'; icon = '⚖️'; aciklama = 'Ağırlık ölçümü'
    elif 'CR' in col_upper:
        birim = '-'; icon = '📈'; aciklama = 'CR değeri'
    elif 'ÇAPI' in col_upper or 'DIAMETER' in col_upper or 'CAP' in col_upper:
        birim = 'mm'; icon = '📏'; aciklama = 'Çap ölçümü'
    elif 'UZUNLUK' in col_upper or 'LENGTH' in col_upper:
        birim = 'm'; icon = '📐'; aciklama = 'Uzunluk ölçümü'
    elif 'SICAKLIK' in col_upper or 'TEMP' in col_upper:
        birim = '°C'; icon = '🌡️'; aciklama = 'Sıcaklık ölçümü'
    elif 'BASINÇ' in col_upper or 'PRESSURE' in col_upper:
        birim = 'bar'; icon = '💨'; aciklama = 'Basınç ölçümü'
    elif 'HIZ' in col_upper or 'SPEED' in col_upper:
        birim = 'm/s'; icon = '⚡'; aciklama = 'Hız ölçümü'
    elif 'FARK' in col_upper or 'DIFF' in col_upper:
        birim = '-'; icon = '📐'; aciklama = 'Fark değeri'

    return {
        'display_name': col_display,
        'sutun': col_name,
        'aciklama': aciklama,
        'birim': birim,
        'icon': icon,
        'predefined': False
    }

def calculate_spc_metrics(data, usl=None, lsl=None):
    n = len(data)
    if n < 2:
        return None

    mean = data.mean()
    std_sample = data.std(ddof=1)

    mr = data.diff().abs().dropna()
    mr_mean = mr.mean() if len(mr) > 0 else 0
    sigma_within = mr_mean / 1.128 if mr_mean > 0 else std_sample

    if sigma_within is None or np.isnan(sigma_within) or sigma_within == 0:
        sigma_within = std_sample if std_sample and std_sample > 0 else 0.0

    ucl = mean + 3 * sigma_within
    lcl = mean - 3 * sigma_within
    mr_ucl = 3.267 * mr_mean if mr_mean > 0 else 0

    sigma_bands = {
        '1sigma_upper': mean + sigma_within,
        '1sigma_lower': mean - sigma_within,
        '2sigma_upper': mean + 2 * sigma_within,
        '2sigma_lower': mean - 2 * sigma_within,
    }

    cp = cpk = cpu = cpl = None
    pp = ppk = ppu = ppl = None
    ppm = ppm_upper = ppm_lower = None
    sigma_level = None

    if usl is not None and lsl is not None and sigma_within > 0:
        cp = (usl - lsl) / (6 * sigma_within)
        cpu = (usl - mean) / (3 * sigma_within)
        cpl = (mean - lsl) / (3 * sigma_within)
        cpk = min(cpu, cpl)

        if std_sample and std_sample > 0:
            pp = (usl - lsl) / (6 * std_sample)
            ppu = (usl - mean) / (3 * std_sample)
            ppl = (mean - lsl) / (3 * std_sample)
            ppk = min(ppu, ppl)

        z_upper = (usl - mean) / sigma_within
        z_lower = (mean - lsl) / sigma_within
        ppm_upper = (1 - stats.norm.cdf(z_upper)) * 1e6
        ppm_lower = stats.norm.cdf(-z_lower) * 1e6
        ppm = ppm_upper + ppm_lower

        if cpk is not None and cpk > 0:
            sigma_level = cpk * 3

    normality_stat, normality_p = None, None
    if 3 <= n <= 5000:
        try:
            normality_stat, normality_p = stats.shapiro(data)
        except:
            pass

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
        'cp': cp, 'cpk': cpk, 'cpu': cpu, 'cpl': cpl,
        'pp': pp, 'ppk': ppk, 'ppu': ppu, 'ppl': ppl,
        'ppm': ppm, 'ppm_upper': ppm_upper, 'ppm_lower': ppm_lower,
        'sigma_level': sigma_level,
        'normality_stat': normality_stat, 'normality_p': normality_p,
        'skewness': skewness, 'kurtosis': kurtosis,
        'n': n,
        'min': data.min(), 'max': data.max(),
        'median': data.median(),
        'q1': data.quantile(0.25), 'q3': data.quantile(0.75),
        'iqr': data.quantile(0.75) - data.quantile(0.25)
    }

def check_western_electric_rules(data, mean, sigma):
    violations = []
    n = len(data)
    if n < 8 or sigma is None or sigma == 0:
        return violations, []

    rule1 = [i for i, val in enumerate(data) if abs(val - mean) > 3 * sigma]
    if rule1:
        violations.append({'rule': 1, 'desc': '3σ dışında nokta', 'points': rule1, 'severity': 'high'})

    rule2 = []
    for i in range(2, n):
        window = data.iloc[i-2:i+1]
        above = ((window - mean) > 2 * sigma).sum()
        below = ((mean - window) > 2 * sigma).sum()
        if above >= 2 or below >= 2:
            rule2.append(i)
    if rule2:
        violations.append({'rule': 2, 'desc': "3 noktadan 2'si 2σ-3σ bölgesinde", 'points': rule2, 'severity': 'medium'})

    rule3 = []
    for i in range(4, n):
        window = data.iloc[i-4:i+1]
        above = ((window - mean) > sigma).sum()
        below = ((mean - window) > sigma).sum()
        if above >= 4 or below >= 4:
            rule3.append(i)
    if rule3:
        violations.append({'rule': 3, 'desc': "5 noktadan 4'ü 1σ-2σ bölgesinde", 'points': rule3, 'severity': 'medium'})

    rule4 = []
    for i in range(7, n):
        window = data.iloc[i-7:i+1]
        if all(window > mean) or all(window < mean):
            rule4.append(i)
    if rule4:
        violations.append({'rule': 4, 'desc': "8 ardışık nokta aynı tarafta", 'points': rule4, 'severity': 'medium'})

    rule5 = []
    for i in range(5, n):
        window = data.iloc[i-5:i+1].values
        diffs = np.diff(window)
        if all(diffs > 0) or all(diffs < 0):
            rule5.append(i)
    if rule5:
        violations.append({'rule': 5, 'desc': "6 ardışık nokta trend oluşturmuş", 'points': rule5, 'severity': 'medium'})

    all_points = set()
    for v in violations:
        all_points.update(v['points'])
    return violations, list(all_points)

def get_cpk_info(cpk):
    if cpk is None:
        return {'color': COLORS['gray'], 'status': 'Belirsiz', 'icon': '❓', 'class': '',
                'desc': 'Tolerans limitleri girilmedi', 'action': 'USL ve LSL değerlerini girin'}
    if cpk >= 1.67:
        return {'color': COLORS['success'], 'status': 'Mükemmel', 'icon': '🌟', 'class': 'excellent',
                'desc': '6σ seviyesine yakın, dünya standartlarında', 'action': 'Mükemmelliği sürdürün'}
    if cpk >= 1.33:
        return {'color': '#22c55e', 'status': 'İyi', 'icon': '✅', 'class': 'good',
                'desc': 'Süreç yeterli, hedeflere uygun', 'action': 'İzlemeye devam edin'}
    if cpk >= 1.0:
        return {'color': COLORS['warning'], 'status': 'Kabul Edilebilir', 'icon': '⚠️', 'class': 'warning',
                'desc': 'Süreç minimum gereksinimleri karşılıyor', 'action': 'İyileştirme fırsatları araştırın'}
    return {'color': COLORS['danger'], 'status': 'Yetersiz', 'icon': '❌', 'class': 'danger',
            'desc': 'Süreç yeterli değil, hata oranı yüksek', 'action': 'ACİL iyileştirme gerekli!'}

def format_number(val, decimals=4):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '-'
    return f"{val:,.{decimals}f}"

def count_out_of_limits(data, ucl, lcl):
    count = 0
    for val in data:
        if ucl is not None and val > ucl:
            count += 1
        if lcl is not None and val < lcl:
            count += 1
    return count

def create_control_chart(data, x_axis, metrics, usl, lsl, title, violation_points=None, param_name="", unit=""):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_axis, y=data,
        mode='lines+markers',
        name='Ölçüm',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=6, color=COLORS['primary']),
        hovertemplate=f'<b>Değer:</b> %{{y:.4f}} {unit}<extra></extra>'
    ))

    for band, opacity in [('2sigma', 0.1), ('1sigma', 0.15)]:
        fig.add_hrect(
            y0=metrics['sigma_bands'][f'{band}_lower'],
            y1=metrics['sigma_bands'][f'{band}_upper'],
            fillcolor=COLORS['primary'],
            opacity=opacity,
            line_width=0
        )

    fig.add_hline(y=metrics['mean'], line_color=COLORS['success'], line_width=2,
                  annotation_text=f"X̄ = {metrics['mean']:.4f}", annotation_position="left")

    fig.add_hline(y=metrics['ucl'], line_color=COLORS['danger'], line_dash='dash', line_width=2,
                  annotation_text=f"UCL = {metrics['ucl']:.4f}", annotation_position="left")
    fig.add_hline(y=metrics['lcl'], line_color=COLORS['danger'], line_dash='dash', line_width=2,
                  annotation_text=f"LCL = {metrics['lcl']:.4f}", annotation_position="left")

    # ✅ 0 değerinde bozulmasın diye is not None
    if usl is not None:
        fig.add_hline(y=usl, line_color=COLORS['warning'], line_dash='dot', line_width=2,
                      annotation_text=f"USL = {usl:.4f}", annotation_position="right")
    if lsl is not None:
        fig.add_hline(y=lsl, line_color=COLORS['warning'], line_dash='dot', line_width=2,
                      annotation_text=f"LSL = {lsl:.4f}", annotation_position="right")

    # Kontrol dışı
    out_mask = (data > metrics['ucl']) | (data < metrics['lcl'])
    if out_mask.any():
        out_points = data[out_mask]
        if isinstance(x_axis, (pd.Series, pd.Index)):
            out_x = x_axis.loc[out_points.index]
        else:
            mask = out_mask.to_numpy()
            out_x = [x_axis[i] for i, m in enumerate(mask) if m]

        fig.add_trace(go.Scatter(
            x=out_x, y=out_points,
            mode='markers',
            name='Kontrol Dışı',
            marker=dict(size=12, color=COLORS['danger'], symbol='x'),
            hovertemplate='<b>KONTROL DIŞI!</b><br>Değer: %{y:.4f}<extra></extra>'
        ))

    # Kural ihlali
    if violation_points:
        valid_vps = [vp for vp in violation_points if vp < len(data)]
        if valid_vps:
            # violation_points enum ile geldiği için iloc kullanmak daha güvenli
            vx = [x_axis.iloc[i] if hasattr(x_axis, 'iloc') else x_axis[i] for i in valid_vps]
            vy = [data.iloc[i] for i in valid_vps]
            fig.add_trace(go.Scatter(
                x=vx, y=vy,
                mode='markers',
                name='Kural İhlali',
                marker=dict(size=10, color=COLORS['purple'], symbol='diamond'),
                hovertemplate='<b>KURAL İHLALİ</b><br>Değer: %{y:.4f}<extra></extra>'
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#1e3a5f')),
        xaxis_title="Zaman / Sıra",
        yaxis_title=f"{param_name} ({unit})" if unit else param_name,
        height=500,
        template="plotly_white",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=12)
    )
    return fig

def create_capability_histogram(data, mean, sigma, usl, lsl, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=30,
        name='Dağılım',
        marker_color=COLORS['light_blue'],
        opacity=0.7
    ))

    # ✅ sigma 0 ise normal eğri çizme
    if sigma is not None and sigma > 0:
        x_range = np.linspace(data.min() - sigma, data.max() + sigma, 100)
        y_norm = stats.norm.pdf(x_range, mean, sigma) * len(data) * (data.max() - data.min()) / 30
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_norm,
            mode='lines',
            name='Normal Dağılım',
            line=dict(color=COLORS['primary'], width=3)
        ))

    fig.add_vline(x=mean, line_color=COLORS['success'], line_width=2,
                  annotation_text=f"X̄={mean:.4f}", annotation_position="top")

    if usl is not None:
        fig.add_vline(x=usl, line_color=COLORS['danger'], line_dash='dash',
                      annotation_text=f"USL={usl:.4f}", annotation_position="top")
    if lsl is not None:
        fig.add_vline(x=lsl, line_color=COLORS['danger'], line_dash='dash',
                      annotation_text=f"LSL={lsl:.4f}", annotation_position="top")

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#1e3a5f')),
        xaxis_title="Değer",
        yaxis_title="Frekans",
        height=400,
        template="plotly_white",
        showlegend=True,
        font=dict(size=12)
    )
    return fig

def create_mr_chart(data, x_axis, mr_mean, mr_ucl, title):
    mr = data.diff().abs().dropna()
    fig = go.Figure()

    # ✅ x eksenini MR index ile hizala
    if isinstance(x_axis, (pd.Series, pd.Index)):
        mr_x = x_axis.loc[mr.index]
    else:
        # liste ise 1'den başlar
        mr_x = x_axis[1:len(mr)+1]

    fig.add_trace(go.Scatter(
        x=mr_x, y=mr,
        mode='lines+markers',
        name='MR',
        line=dict(color=COLORS['purple'], width=2),
        marker=dict(size=6)
    ))

    fig.add_hline(y=mr_mean, line_color=COLORS['success'], line_width=2,
                  annotation_text=f"MR̄ = {mr_mean:.4f}", annotation_position="left")
    fig.add_hline(y=mr_ucl, line_color=COLORS['danger'], line_dash='dash',
                  annotation_text=f"UCL = {mr_ucl:.4f}", annotation_position="left")

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#1e3a5f')),
        xaxis_title="Zaman / Sıra",
        yaxis_title="Moving Range",
        height=350,
        template="plotly_white",
        font=dict(size=12)
    )
    return fig

def create_cpk_gauge(cpk):
    cpk_info = get_cpk_info(cpk)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=cpk if cpk is not None else 0,
        number={'suffix': "", 'font': {'size': 40}},
        title={'text': f"Cpk<br><span style='font-size:0.7em;color:{cpk_info['color']}'>{cpk_info['status']}</span>"},
        delta={'reference': 1.33, 'increasing': {'color': COLORS['success']}},
        gauge={
            'axis': {'range': [0, 2.5], 'tickwidth': 1},
            'bar': {'color': cpk_info['color'], 'thickness': 0.75},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': '#e2e8f0',
            'steps': [
                {'range': [0, 1.0], 'color': COLORS['light_red']},
                {'range': [1.0, 1.33], 'color': '#fef3c7'},
                {'range': [1.33, 1.67], 'color': COLORS['light_green']},
                {'range': [1.67, 2.5], 'color': '#86efac'}
            ],
            'threshold': {'line': {'color': COLORS['danger'], 'width': 3}, 'thickness': 0.75, 'value': 1.33}
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20), font=dict(size=14))
    return fig

def parse_float_optional(txt: str):
    txt = (txt or "").strip().replace(",", ".")
    if txt == "":
        return None
    try:
        return float(txt)
    except:
        return None

# ============================================
# VERİ YÜKLEME
# ============================================
st.sidebar.markdown(f"### 📊 {SIRKET_ISMI}")
st.sidebar.markdown("---")

if os.path.exists(LOGO_DOSYA_ADI):
    st.sidebar.image(LOGO_DOSYA_ADI, width=200)

df = None
error = None

# ✅ Sabit dosya yoksa klasörden en güncel SPC dosyasını seç
file_to_try = SABIT_DOSYA_ADI if os.path.exists(SABIT_DOSYA_ADI) else pick_latest_spc_file(KLASOR_YOLU)

if file_to_try and os.path.exists(file_to_try):
    try:
        df = pd.read_excel(file_to_try)
        df = normalize_columns(df)
        st.sidebar.success(f"✅ Yüklendi: {os.path.basename(file_to_try)}")
    except Exception as e:
        error = str(e)
        st.sidebar.error(f"Hata: {error}")
else:
    uploaded = st.sidebar.file_uploader("📁 Veri Yükle", type=['xlsx', 'csv'])
    if uploaded:
        if uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        df = normalize_columns(df)

if df is None:
    st.markdown('<p class="main-header">📊 Gelişmiş SPC Analiz Sistemi</p>', unsafe_allow_html=True)
    st.info("📁 Lütfen veri dosyası yükleyin veya SPC*.xlsx dosyasını klasöre ekleyin.")
    st.stop()

# ============================================
# VERİ HAZIRLAMA
# ============================================
if COL_DATE in df.columns:
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors='coerce')

# Direnç farkı hesapla (varsa)
direnc_baslangic_col = None
direnc_bitis_col = None
for col in df.columns:
    col_norm = str(col).upper()
    if 'KALİTE BAŞLANGIÇ' in col_norm and 'DİRENÇ' in col_norm:
        direnc_baslangic_col = col
    if 'KALİTE BİTİŞ' in col_norm and 'DİRENÇ' in col_norm:
        direnc_bitis_col = col

if direnc_baslangic_col and direnc_bitis_col:
    baslangic_numeric = convert_column_to_numeric(df, direnc_baslangic_col)
    bitis_numeric = convert_column_to_numeric(df, direnc_bitis_col)
    df['DIRENC_FARKI'] = baslangic_numeric - bitis_numeric

df_original = df.copy()

# ============================================
# FİLTRELER
# ============================================
st.sidebar.markdown("### 🔍 Filtreler")

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

secili_kesit = 'Tümü'
if COL_GROUP in df.columns:
    kesitler = ['Tümü'] + sorted(df[COL_GROUP].dropna().unique().tolist())
    secili_kesit = st.sidebar.selectbox("📦 Kesit", kesitler)
    if secili_kesit != 'Tümü':
        df = df[df[COL_GROUP] == secili_kesit]

secili_makine = 'Tümü'
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

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">📊 Toplam Kayıt</div>
    <div class="metric-value">{len(df):,}</div></div>""", unsafe_allow_html=True)
with col2:
    kesit_sayi = df[COL_GROUP].nunique() if COL_GROUP in df.columns else 0
    st.markdown(f"""<div class="metric-card"><div class="metric-label">📦 Kesit Çeşidi</div>
    <div class="metric-value">{kesit_sayi}</div></div>""", unsafe_allow_html=True)
with col3:
    makine_sayi = df[COL_MACHINE].nunique() if COL_MACHINE in df.columns else 0
    st.markdown(f"""<div class="metric-card"><div class="metric-label">🏭 Makine Sayısı</div>
    <div class="metric-value">{makine_sayi}</div></div>""", unsafe_allow_html=True)
with col4:
    if COL_DATE in df.columns and len(df) > 0 and df[COL_DATE].notna().any():
        gun = (df[COL_DATE].max() - df[COL_DATE].min()).days + 1
    else:
        gun = 0
    st.markdown(f"""<div class="metric-card"><div class="metric-label">📅 Analiz Süresi</div>
    <div class="metric-value">{gun} gün</div></div>""", unsafe_allow_html=True)

# ============================================
# PARAMETRE SEÇİMİ
# ============================================
st.markdown('<p class="section-header">🔬 Parametre Seçimi</p>', unsafe_allow_html=True)

available_numeric_cols = get_available_numeric_columns(df)
if not available_numeric_cols:
    st.error("❌ Analiz edilebilecek sayısal sütun bulunamadı!")
    st.stop()

column_options = {}
for col in available_numeric_cols:
    col_info = get_column_info(col, df)
    display_name = col_info['display_name']
    if display_name in column_options:
        display_name = f"{display_name} [{str(col)[:20]}...]"
    column_options[display_name] = col_info

col_select, col_info_display = st.columns([2, 3])
with col_select:
    predefined = [k for k, v in column_options.items() if v['predefined']]
    dynamic = [k for k, v in column_options.items() if not v['predefined']]
    all_options = predefined + (['─' * 20] if predefined and dynamic else []) + dynamic

    selected_display = st.selectbox(
        "📈 Analiz edilecek parametre:",
        [opt for opt in all_options if opt != '─' * 20],
        help="Excel dosyasındaki tüm sayısal sütunlar listelenir"
    )

param_info = column_options[selected_display]
secili_sutun = param_info['sutun']

with col_info_display:
    predefined_badge = (
        '<span style="background:#16a34a;color:white;padding:2px 8px;border-radius:4px;font-size:0.75rem;margin-left:8px;">Tanımlı</span>'
        if param_info['predefined']
        else '<span style="background:#6b7280;color:white;padding:2px 8px;border-radius:4px;font-size:0.75rem;margin-left:8px;">Dinamik</span>'
    )
    st.markdown(f"""
    <div class="info-box">
        <b>{param_info['icon']} {param_info['display_name']}</b> {predefined_badge}<br>
        {param_info['aciklama']}<br>
        <small><b>Birim:</b> {param_info['birim']} | <b>Sütun:</b> {str(param_info['sutun'])}</small>
    </div>
    """, unsafe_allow_html=True)

with st.expander("📋 Dosyadaki Tüm Sayısal Sütunlar", expanded=False):
    cols_data = []
    for col in available_numeric_cols:
        info = get_column_info(col, df)
        numeric_col = convert_column_to_numeric(df, col)
        cols_data.append({
            'Sütun Adı': str(col),
            'Görünen Ad': info['display_name'],
            'Birim': info['birim'],
            'Tip': 'Tanımlı' if info['predefined'] else 'Dinamik',
            'Veri Sayısı': numeric_col.notna().sum(),
            'Min': f"{numeric_col.min():.4f}" if pd.notna(numeric_col.min()) else '-',
            'Max': f"{numeric_col.max():.4f}" if pd.notna(numeric_col.max()) else '-'
        })
    st.dataframe(pd.DataFrame(cols_data), use_container_width=True, hide_index=True)

# ============================================
# TOLERANS (✅ value=None HATASI FIX)
# ============================================
col_tol1, col_tol2, col_tol3 = st.columns([1, 1, 1])
with col_tol1:
    use_spec = st.checkbox("📏 Tolerans Limitleri Kullan", value=False)

usl, lsl = None, None
if use_spec:
    with col_tol2:
        usl_txt = st.text_input("USL (Üst Tolerans)", value="", placeholder="örn: 1.2345")
    with col_tol3:
        lsl_txt = st.text_input("LSL (Alt Tolerans)", value="", placeholder="örn: 0.9876")

    usl = parse_float_optional(usl_txt)
    lsl = parse_float_optional(lsl_txt)

    if (usl_txt.strip() and usl is None) or (lsl_txt.strip() and lsl is None):
        st.sidebar.error("USL/LSL sayı olmalı. Örn: 1.2345 veya 1,2345")
    if usl is not None and lsl is not None and usl <= lsl:
        st.sidebar.error("USL, LSL'den büyük olmalı!")

# ============================================
# HESAPLAMA
# ============================================
if COL_DATE in df.columns:
    df = df.sort_values(COL_DATE)

data = convert_column_to_numeric(df, secili_sutun).dropna()
if len(data) < 2:
    st.warning("⚠️ Yeterli veri yok (en az 2 ölçüm gerekli)")
    st.stop()

metrics = calculate_spc_metrics(data, usl, lsl)
if metrics is None:
    st.error("❌ SPC hesaplamaları yapılamadı!")
    st.stop()

violations, violation_points = check_western_electric_rules(data, metrics['mean'], metrics['sigma_within'])
cpk_info = get_cpk_info(metrics['cpk'])

# ============================================
# ANA METRİKLER
# ============================================
st.markdown('<p class="section-header">📊 Temel Metrikler</p>', unsafe_allow_html=True)
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

with col_m1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">📊 Veri Sayısı</div>
    <div class="metric-value">{metrics['n']:,}</div></div>""", unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">📈 Ortalama (X̄)</div>
    <div class="metric-value">{format_number(metrics['mean'])}</div></div>""", unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">📉 Std Sapma (σ)</div>
    <div class="metric-value">{format_number(metrics['sigma_within'])}</div></div>""", unsafe_allow_html=True)
with col_m4:
    out_count = count_out_of_limits(data, metrics['ucl'], metrics['lcl'])
    out_class = 'danger' if out_count > 0 else 'excellent'
    st.markdown(f"""<div class="metric-card {out_class}"><div class="metric-label">⚠️ Kontrol Dışı</div>
    <div class="metric-value">{out_count}</div><div class="metric-desc">{out_count/len(data)*100:.1f}% oran</div></div>""",
                unsafe_allow_html=True)
with col_m5:
    violation_class = 'danger' if len(violations) > 0 else 'excellent'
    st.markdown(f"""<div class="metric-card {violation_class}"><div class="metric-label">📏 Kural İhlali</div>
    <div class="metric-value">{len(violations)}</div><div class="metric-desc">Western Electric</div></div>""",
                unsafe_allow_html=True)

# ============================================
# YETERLİLİK
# ============================================
if metrics['cpk'] is not None:
    st.markdown('<p class="section-header">⭐ Yeterlilik Analizi</p>', unsafe_allow_html=True)
    col_gauge, col_indices = st.columns([1, 2])

    with col_gauge:
        st.plotly_chart(create_cpk_gauge(metrics['cpk']), use_container_width=True)
        st.markdown(f"""
        <div class="alert-box alert-{'success' if cpk_info['class'] in ['excellent','good'] else 'warning' if cpk_info['class']=='warning' else 'danger'}">
            <b>{cpk_info['icon']} {cpk_info['status']}</b><br>
            {cpk_info['desc']}<br>
            <small><b>Aksiyon:</b> {cpk_info['action']}</small>
        </div>
        """, unsafe_allow_html=True)

    with col_indices:
        col_idx1, col_idx2 = st.columns(2)
        with col_idx1:
            st.markdown(f"""<div class="metric-card {'excellent' if metrics['cp'] and metrics['cp'] >= 1.33 else 'warning'}">
            <div class="metric-label">Cp (Potansiyel)</div><div class="metric-value">{format_number(metrics['cp'], 2)}</div>
            <div class="metric-desc">Süreç potansiyeli</div></div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="metric-card {'excellent' if metrics['pp'] and metrics['pp'] >= 1.33 else 'warning'}">
            <div class="metric-label">Pp (Uzun Vadeli)</div><div class="metric-value">{format_number(metrics['pp'], 2)}</div>
            <div class="metric-desc">Genel performans</div></div>""", unsafe_allow_html=True)

        with col_idx2:
            st.markdown(f"""<div class="metric-card {cpk_info['class']}">
            <div class="metric-label">Cpk (Performans)</div><div class="metric-value">{format_number(metrics['cpk'], 2)}</div>
            <div class="metric-desc">Kısa vadeli yetenek</div></div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="metric-card {'excellent' if metrics['ppk'] and metrics['ppk'] >= 1.33 else 'warning'}">
            <div class="metric-label">Ppk (Uzun Vadeli)</div><div class="metric-value">{format_number(metrics['ppk'], 2)}</div>
            <div class="metric-desc">Uzun vadeli yetenek</div></div>""", unsafe_allow_html=True)

# ============================================
# NORMALLİK
# ============================================
if metrics['normality_p'] is not None:
    with st.expander("📈 Normallik Testi (Shapiro-Wilk)", expanded=False):
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            is_normal = metrics['normality_p'] > 0.05
            st.markdown(f"""<div class="metric-card {'excellent' if is_normal else 'warning'}">
            <div class="metric-label">p-değeri</div><div class="metric-value">{metrics['normality_p']:.4f}</div>
            <div class="metric-desc">{'✅ Normal dağılım (p > 0.05)' if is_normal else '⚠️ Normal değil (p ≤ 0.05)'}</div></div>""",
                        unsafe_allow_html=True)
        with col_n2:
            st.markdown(f"""
            **Çarpıklık (Skewness):** {metrics['skewness']:.3f}  
            **Basıklık (Kurtosis):** {metrics['kurtosis']:.3f}
            """)
        if not is_normal:
            st.warning("⚠️ Veriler normal dağılmıyor. Cp/Cpk değerlerini dikkatli yorumlayın (gerekirse dönüşüm).")

# ============================================
# WESTERN ELECTRIC
# ============================================
if len(data) >= 8:
    with st.expander("📏 Western Electric Kuralları", expanded=len(violations) > 0):
        if not violations:
            st.markdown("""<div class="rule-ok">✅ <b>Tüm kurallar geçti!</b> Süreç kontrol altında.</div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="rule-violation">⚠️ <b>{len(violations)} kural ihlali tespit edildi!</b></div>""",
                        unsafe_allow_html=True)
            for v in violations:
                severity_icon = "🔴" if v['severity'] == 'high' else "🟡"
                st.markdown(f"""<div class="rule-violation">
                {severity_icon} <b>Kural {v['rule']}:</b> {v['desc']}<br>
                <small>İhlal noktası sayısı: {len(v['points'])}</small></div>""", unsafe_allow_html=True)

# ============================================
# GRAFİKLER
# ============================================
st.markdown('<p class="section-header">📈 Grafikler</p>', unsafe_allow_html=True)

if COL_DATE in df.columns and df[COL_DATE].notna().any():
    x_axis = df.loc[data.index, COL_DATE]
else:
    x_axis = list(range(len(data)))

tab1, tab2, tab3 = st.tabs(["📈 Kontrol Grafiği", "📊 Histogram", "📉 Moving Range"])

with tab1:
    control_fig = create_control_chart(
        data, x_axis, metrics, usl, lsl,
        title=f"I-MR Kontrol Grafiği - {param_info['display_name']}",
        violation_points=violation_points,
        param_name=param_info['display_name'],
        unit=param_info['birim']
    )
    st.plotly_chart(control_fig, use_container_width=True)

with tab2:
    hist_fig = create_capability_histogram(
        data, metrics['mean'], metrics['sigma_within'], usl, lsl,
        title=f"Yetenek Histogramı - {param_info['display_name']}"
    )
    st.plotly_chart(hist_fig, use_container_width=True)

with tab3:
    mr_fig = create_mr_chart(
        data, x_axis, metrics['mr_mean'], metrics['mr_ucl'],
        title="Moving Range (MR) Grafiği"
    )
    st.plotly_chart(mr_fig, use_container_width=True)

    mr_series = data.diff().abs().dropna()
    mr_out_count = count_out_of_limits(mr_series, metrics['mr_ucl'], None)
    if mr_out_count > 0:
        st.warning(f"⚠️ {mr_out_count} adet MR değeri kontrol limitinin üzerinde (ani değişimler var).")

# ============================================
# MAKİNE KARŞILAŞTIRMA
# ============================================
if COL_MACHINE in df.columns and df[COL_MACHINE].nunique() > 1:
    st.markdown('<p class="section-header">🏭 Makine Karşılaştırma</p>', unsafe_allow_html=True)

    df_machine = df.copy()
    df_machine['_numeric_col'] = convert_column_to_numeric(df_machine, secili_sutun)

    machine_stats = df_machine.groupby(COL_MACHINE)['_numeric_col'].agg(['mean', 'std', 'count', 'min', 'max']).round(4)
    machine_stats.columns = ['Ortalama', 'Std Sapma', 'Kayıt', 'Min', 'Max']
    machine_stats = machine_stats.sort_values('Kayıt', ascending=False)

    col_table, col_box = st.columns([1, 2])
    with col_table:
        st.dataframe(machine_stats, use_container_width=True, height=400)

    with col_box:
        fig_box = px.box(
            df_machine, x=COL_MACHINE, y='_numeric_col',
            color=COL_MACHINE,
            title=f"Makine Bazlı {param_info['display_name']} Dağılımı"
        )
        fig_box.add_hline(y=metrics['mean'], line_color=COLORS['success'], line_dash='dash', annotation_text="Genel Ort.")
        if usl is not None:
            fig_box.add_hline(y=usl, line_color=COLORS['warning'], annotation_text="USL")
        if lsl is not None:
            fig_box.add_hline(y=lsl, line_color=COLORS['warning'], annotation_text="LSL")
        fig_box.update_layout(height=450, template="plotly_white", showlegend=False, font=dict(size=12))
        st.plotly_chart(fig_box, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"""
<center><small>
📊 <b>Gelişmiş SPC Analiz Sistemi (fix)</b> | {SIRKET_ISMI} |
Analiz: <b>{len(data):,}</b> kayıt |
{f"Kesit: <b>{secili_kesit}</b> | " if secili_kesit != 'Tümü' else ""}
Parametre: <b>{param_info['display_name']}</b>
</small></center>
""", unsafe_allow_html=True)
