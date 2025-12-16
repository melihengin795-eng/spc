import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
    # --- YENİ EKLENEN PARAMETRE ---
    'Başlangıç Birim Ağırlık (g/m)': {
        'sutun': 'BAŞLANGIÇ BİRİM AĞIRLIK',
        'aciklama': 'Üretim başlangıcında ölçülen birim ağırlık değeri.',
        'birim': 'g/m',
        'icon': '⚖️'
    },
    # ------------------------------
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
        'sutun': 'DIRENC_FARKI',  # Hesaplanacak
        'aciklama': 'Başlangıç ve bitiş direnci arasındaki fark. Üretim sürecindeki direnç değişimini gösterir.',
        'birim': 'Ω (Ohm)',
        'icon': '📐'
    },
    'CR Farkı (Başlangıç-Bitiş)': {
        'sutun': 'BAŞLANGIÇ - BİTİŞ CR',
        'aciklama': 'Başlangıç ve bitiş CR değerleri arasındaki fark. Süreç stabilitesini gösterir.',
        'birim': '-',
        'icon': '📊'
    },
    'Başlangıç CR': {
        'sutun': 'BAŞLANGIÇ CR',
        'aciklama': 'Üretim başlangıcındaki CR (Conductor Resistance) değeri.',
        'birim': '-',
        'icon': '📈'
    },
    'Bitiş CR': {
        'sutun': 'BİTİŞ CR',
        'aciklama': 'Üretim bitişindeki CR (Conductor Resistance) değeri.',
        'birim': '-',
        'icon': '📉'
    }
}

# ============================================
# CSS STİLLERİ
# ============================================
st.markdown("""
<style>
    /* Ana başlıklar */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding: 0.5rem 0;
        border-left: 4px solid #3498db;
        padding-left: 1rem;
    }
    
    /* Metrik kutuları */
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        text-align: center;
        margin: 0.75rem 0;
        transition: transform 0.2s;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1e3a5f;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #666;
        font-weight: 500;
    }
    
    .metric-desc {
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.5rem;
    }
    
    /* Durum renkleri */
    .status-excellent { border-left: 5px solid #27ae60; background: linear-gradient(to right, #e8f8f0, white); }
    .status-good { border-left: 5px solid #2ecc71; }
    .status-warning { border-left: 5px solid #f39c12; background: linear-gradient(to right, #fef9e7, white); }
    .status-bad { border-left: 5px solid #e74c3c; background: linear-gradient(to right, #fdedec, white); }
    
    /* Bilgi kutuları */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .param-info {
        background: #f8f9fa;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    
    .param-info h4 {
        color: #2c3e50;
        margin: 0 0 0.5rem 0;
    }
    
    .param-info p {
        color: #666;
        margin: 0;
        font-size: 0.95rem;
    }
    
    /* Açıklama kartları */
    .explain-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 0.75rem 0;
    }
    
    .explain-title {
        font-weight: 600;
        color: #2c3e50;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .explain-formula {
        background: #ecf0f1;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        margin: 0.5rem 0;
        display: inline-block;
    }
    
    /* Tablo stilleri */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    
    .styled-table th {
        background: #3498db;
        color: white;
        padding: 12px;
        text-align: left;
    }
    
    .styled-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #eee;
    }
    
    /* Grafik container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin: 1.5rem 0;
    }
    
    /* Spacer */
    .spacer { margin: 2rem 0; }
    .spacer-sm { margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================
# YARDIMCI FONKSİYONLAR
# ============================================

def calculate_spc_metrics(data, usl=None, lsl=None):
    """Tüm SPC metriklerini hesapla"""
    n = len(data)
    if n < 2:
        return None
    
    mean = data.mean()
    std_sample = data.std()
    
    # Moving Range ile sigma tahmini
    mr = data.diff().abs()
    mr_mean = mr.mean()
    sigma_mr = mr_mean / 1.128 if mr_mean > 0 else std_sample
    
    # Kontrol limitleri
    ucl = mean + 3 * sigma_mr
    lcl = mean - 3 * sigma_mr
    mr_ucl = 3.267 * mr_mean if mr_mean > 0 else 0
    
    # Yeterlilik indeksleri
    cp, cpk, pp, ppk, ppm = None, None, None, None, None
    
    if usl is not None and lsl is not None and sigma_mr > 0:
        cp = (usl - lsl) / (6 * sigma_mr)
        cpu = (usl - mean) / (3 * sigma_mr)
        cpl = (mean - lsl) / (3 * sigma_mr)
        cpk = min(cpu, cpl)
        
        if std_sample > 0:
            pp = (usl - lsl) / (6 * std_sample)
            ppu = (usl - mean) / (3 * std_sample)
            ppl = (mean - lsl) / (3 * std_sample)
            ppk = min(ppu, ppl)
        
        ppm_upper = (1 - stats.norm.cdf((usl - mean) / sigma_mr)) * 1e6
        ppm_lower = stats.norm.cdf((lsl - mean) / sigma_mr) * 1e6
        ppm = ppm_upper + ppm_lower
    
    return {
        'mean': mean,
        'std': std_sample,
        'sigma_mr': sigma_mr,
        'ucl': ucl,
        'lcl': lcl,
        'mr_mean': mr_mean,
        'mr_ucl': mr_ucl,
        'cp': cp,
        'cpk': cpk,
        'pp': pp,
        'ppk': ppk,
        'ppm': ppm,
        'n': n,
        'min': data.min(),
        'max': data.max(),
        'median': data.median(),
        'q1': data.quantile(0.25),
        'q3': data.quantile(0.75)
    }

def get_cpk_status(cpk):
    """Cpk değerine göre durum"""
    if cpk is None:
        return "gray", "Belirsiz", "-", "status-good"
    elif cpk >= 1.67:
        return "#27ae60", "Mükemmel", "✓✓", "status-excellent"
    elif cpk >= 1.33:
        return "#2ecc71", "İyi", "✓", "status-good"
    elif cpk >= 1.00:
        return "#f39c12", "Kabul Edilebilir", "~", "status-warning"
    else:
        return "#e74c3c", "Yetersiz", "✗", "status-bad"

def count_out_of_limits(data, upper, lower):
    """Limit dışı nokta sayısı"""
    count = 0
    if upper is not None:
        count += (data > upper).sum()
    if lower is not None:
        count += (data < lower).sum()
    return count

def format_number(val, decimals=4):
    """Sayı formatlama"""
    if val is None:
        return "-"
    if abs(val) >= 1000:
        return f"{val:,.2f}"
    elif abs(val) >= 1:
        return f"{val:.{decimals}f}"
    else:
        return f"{val:.6f}"

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
st.sidebar.markdown("**SPC Analiz Sistemi**")
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
    st.markdown('<p class="main-header">📊 SPC Analiz Sistemi</p>', unsafe_allow_html=True)
    st.info(f"📁 Lütfen veri dosyası yükleyin veya **{os.path.basename(SABIT_DOSYA_ADI)}** dosyasını klasöre ekleyin.")
    
    # Parametre açıklamaları
    st.markdown('<p class="section-header">📚 SPC Nedir?</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **İstatistiksel Süreç Kontrolü (SPC)**, üretim süreçlerini izlemek ve kontrol altında tutmak için kullanılan 
    istatistiksel yöntemler bütünüdür. Temel amacı:
    
    - 🎯 Süreç değişkenliğini azaltmak
    - ⚠️ Problemleri erken tespit etmek  
    - 📈 Kaliteyi sürekli iyileştirmek
    """)
    
    st.markdown('<p class="section-header">📊 Temel Parametreler</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="explain-card">
            <div class="explain-title">📈 Ortalama (X̄)</div>
            <p>Tüm ölçümlerin aritmetik ortalaması. Sürecin "merkezi"ni gösterir.</p>
            <div class="explain-formula">X̄ = Σx / n</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explain-card">
            <div class="explain-title">📊 Standart Sapma (σ)</div>
            <p>Verilerin ortalamadan ne kadar uzaklaştığını gösterir. Düşük σ = tutarlı süreç.</p>
            <div class="explain-formula">σ = √[Σ(x-X̄)² / (n-1)]</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explain-card">
            <div class="explain-title">🔴 UCL / LCL (Kontrol Limitleri)</div>
            <p>Sürecin doğal sınırları. ±3σ ile hesaplanır. Bu limitler dışındaki noktalar "özel neden" kaynaklıdır.</p>
            <div class="explain-formula">UCL = X̄ + 3σ &nbsp;|&nbsp; LCL = X̄ - 3σ</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="explain-card">
            <div class="explain-title">🟠 USL / LSL (Tolerans Limitleri)</div>
            <p>Müşteri veya mühendislik tarafından belirlenen kabul edilebilir sınırlar.</p>
            <div class="explain-formula">Müşteri Spesifikasyonu</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explain-card">
            <div class="explain-title">⭐ Cp (Süreç Potansiyeli)</div>
            <p>Tolerans genişliğinin süreç yayılımına oranı. Merkezlemeyi dikkate almaz.</p>
            <div class="explain-formula">Cp = (USL - LSL) / 6σ</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explain-card">
            <div class="explain-title">🏆 Cpk (Süreç Yeterliliği)</div>
            <p>Gerçek performans. Merkezlemeyi de hesaba katar. <b>Hedef: Cpk ≥ 1.33</b></p>
            <div class="explain-formula">Cpk = min(CPU, CPL)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📋 Cpk Değerlendirme Tablosu")
    
    cpk_df = pd.DataFrame({
        "Cpk Aralığı": ["< 1.00", "1.00 - 1.33", "1.33 - 1.67", "> 1.67"],
        "Durum": ["❌ Yetersiz", "⚠️ Kabul Edilebilir", "✅ İyi", "🌟 Mükemmel"],
        "Sigma Seviyesi": ["< 3σ", "3σ - 4σ", "4σ - 5σ", "> 5σ"],
        "Tahmini Hata (PPM)": ["> 2,700", "66 - 2,700", "< 66", "< 1"],
        "Aksiyon": ["Acil müdahale", "İyileştirme planla", "İzlemeye devam", "Mükemmelliği sürdür"]
    })
    st.dataframe(cpk_df, use_container_width=True, hide_index=True)
    
    st.stop()

# ============================================
# VERİ HAZIRLAMA
# ============================================
# Tarih düzenleme
if COL_DATE in df.columns:
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors='coerce')

# Direnç farkını hesapla
if 'KALİTE BAŞLANGIÇ ÖLÇÜLEN DİRENÇ' in df.columns and 'KALİTE BİTİŞ ÖLÇÜLEN DİRENÇ' in df.columns:
    df['DIRENC_FARKI'] = df['KALİTE BAŞLANGIÇ ÖLÇÜLEN DİRENÇ'] - df['KALİTE BİTİŞ ÖLÇÜLEN DİRENÇ']

# Orijinal veri
df_original = df.copy()

# ============================================
# FİLTRELER
# ============================================
st.sidebar.markdown("### 🔍 Filtreler")

# Varsayılan değerler
secili_kesit = 'Tümü'
secili_makine = 'Tümü'

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
st.sidebar.metric("📊 Kayıt Sayısı", f"{len(df):,}")

# ============================================
# ANA BAŞLIK
# ============================================
st.markdown(f'<p class="main-header">📊 SPC Analiz - {SIRKET_ISMI}</p>', unsafe_allow_html=True)

# Özet kartları
st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">📊 Toplam Kayıt</div>
        <div class="metric-value">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    kesit_sayi = df[COL_GROUP].nunique() if COL_GROUP in df.columns else 0
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">📦 Kesit Çeşidi</div>
        <div class="metric-value">{kesit_sayi}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    makine_sayi = df[COL_MACHINE].nunique() if COL_MACHINE in df.columns else 0
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">🏭 Makine Sayısı</div>
        <div class="metric-value">{makine_sayi}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    if COL_DATE in df.columns and len(df) > 0:
        gun = (df[COL_DATE].max() - df[COL_DATE].min()).days + 1
    else:
        gun = 0
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">📅 Analiz Süresi</div>
        <div class="metric-value">{gun} gün</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PARAMETRE SEÇİMİ
# ============================================
st.markdown('<p class="section-header">🔬 Parametre Analizi</p>', unsafe_allow_html=True)

# Mevcut parametreleri kontrol et
mevcut_params = {}
for key, val in PARAM_MAP.items():
    if val['sutun'] in df.columns:
        mevcut_params[key] = val

if not mevcut_params:
    st.error("Analiz edilebilecek parametre bulunamadı!")
    st.stop()

# Parametre seçimi
col_select, col_info = st.columns([2, 3])

with col_select:
    secili_param = st.selectbox(
        "📈 Analiz edilecek parametre:",
        list(mevcut_params.keys()),
        help="Analiz yapmak istediğiniz parametreyi seçin"
    )

param_info = mevcut_params[secili_param]

with col_info:
    st.markdown(f"""
    <div class="param-info">
        <h4>{param_info['icon']} {secili_param}</h4>
        <p>{param_info['aciklama']}</p>
        <p><b>Birim:</b> {param_info['birim']}</p>
    </div>
    """, unsafe_allow_html=True)

# Tolerans ayarları
st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

col_tol1, col_tol2, col_tol3 = st.columns([1, 1, 1])

with col_tol1:
    use_spec = st.checkbox("📏 Tolerans Limitleri Kullan", value=False, 
                           help="USL ve LSL değerlerini girerek Cp/Cpk hesaplaması yapabilirsiniz")

usl, lsl = None, None
if use_spec:
    with col_tol2:
        usl = st.number_input("USL (Üst Tolerans)", value=None, format="%.4f",
                              help="Üst Spesifikasyon Limiti - Kabul edilebilir maksimum değer")
    with col_tol3:
        lsl = st.number_input("LSL (Alt Tolerans)", value=None, format="%.4f",
                              help="Alt Spesifikasyon Limiti - Kabul edilebilir minimum değer")

# ============================================
# VERİ HAZIRLAMA VE HESAPLAMA
# ============================================
secili_sutun = param_info['sutun']

# Tarihe göre sırala
if COL_DATE in df.columns:
    df = df.sort_values(COL_DATE)

data = df[secili_sutun].dropna()

if len(data) < 2:
    st.warning("⚠️ Yeterli veri yok (en az 2 ölçüm gerekli)")
    st.stop()

# SPC hesaplamaları
metrics = calculate_spc_metrics(data, usl, lsl)

if metrics is None:
    st.error("Hesaplama hatası!")
    st.stop()

# ============================================
# SONUÇLAR
# ============================================
st.markdown('<p class="section-header">📊 Analiz Sonuçları</p>', unsafe_allow_html=True)

# Ana istatistikler
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">📈 Ortalama (X̄)</div>
        <div class="metric-value">{format_number(metrics['mean'])}</div>
        <div class="metric-desc">Merkezi değer</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">📊 Std Sapma (σ)</div>
        <div class="metric-value">{format_number(metrics['sigma_mr'])}</div>
        <div class="metric-desc">Değişkenlik</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">⬇️ Minimum</div>
        <div class="metric-value">{format_number(metrics['min'])}</div>
        <div class="metric-desc">En düşük</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">⬆️ Maksimum</div>
        <div class="metric-value">{format_number(metrics['max'])}</div>
        <div class="metric-desc">En yüksek</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    out_ctrl = count_out_of_limits(data, metrics['ucl'], metrics['lcl'])
    status = "status-good" if out_ctrl == 0 else "status-bad"
    st.markdown(f"""
    <div class="metric-box {status}">
        <div class="metric-label">🔴 Kontrol Dışı</div>
        <div class="metric-value">{out_ctrl}</div>
        <div class="metric-desc">UCL/LCL dışı</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">📋 Veri Sayısı</div>
        <div class="metric-value">{metrics['n']:,}</div>
        <div class="metric-desc">Toplam ölçüm</div>
    </div>
    """, unsafe_allow_html=True)

# Kontrol limitleri bilgisi
st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

col_ucl, col_lcl, col_range = st.columns(3)

with col_ucl:
    st.markdown(f"""
    <div class="metric-box" style="border-top: 3px solid #e74c3c;">
        <div class="metric-label">🔺 Üst Kontrol Limiti (UCL)</div>
        <div class="metric-value" style="color: #e74c3c;">{format_number(metrics['ucl'])}</div>
        <div class="metric-desc">X̄ + 3σ</div>
    </div>
    """, unsafe_allow_html=True)

with col_lcl:
    st.markdown(f"""
    <div class="metric-box" style="border-top: 3px solid #e74c3c;">
        <div class="metric-label">🔻 Alt Kontrol Limiti (LCL)</div>
        <div class="metric-value" style="color: #e74c3c;">{format_number(metrics['lcl'])}</div>
        <div class="metric-desc">X̄ - 3σ</div>
    </div>
    """, unsafe_allow_html=True)

with col_range:
    range_val = metrics['max'] - metrics['min']
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">↔️ Veri Aralığı (Range)</div>
        <div class="metric-value">{format_number(range_val)}</div>
        <div class="metric-desc">Max - Min</div>
    </div>
    """, unsafe_allow_html=True)

# Yeterlilik (sadece tolerans varsa)
if metrics['cpk'] is not None:
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    st.markdown("#### ⭐ Süreç Yeterliliği")
    
    cpk_color, cpk_status, cpk_icon, cpk_class = get_cpk_status(metrics['cpk'])
    
    col_cp, col_cpk, col_pp, col_ppk, col_ppm = st.columns(5)
    
    with col_cp:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Cp (Potansiyel)</div>
            <div class="metric-value">{format_number(metrics['cp'], 2)}</div>
            <div class="metric-desc">Merkezleme hariç</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_cpk:
        st.markdown(f"""
        <div class="metric-box {cpk_class}">
            <div class="metric-label">Cpk {cpk_icon}</div>
            <div class="metric-value" style="color: {cpk_color};">{format_number(metrics['cpk'], 2)}</div>
            <div class="metric-desc">{cpk_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_pp:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Pp (Uzun Vadeli)</div>
            <div class="metric-value">{format_number(metrics['pp'], 2)}</div>
            <div class="metric-desc">Örnek std ile</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ppk:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Ppk</div>
            <div class="metric-value">{format_number(metrics['ppk'], 2)}</div>
            <div class="metric-desc">Uzun vadeli</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ppm:
        if metrics['ppm'] is not None and metrics['ppm'] < 1e6:
            ppm_val = format_number(metrics['ppm'], 0)
        else:
            ppm_val = "-" if metrics['ppm'] is None else "< 1"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">PPM</div>
            <div class="metric-value">{ppm_val}</div>
            <div class="metric-desc">Milyonda hata</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tolerans dışı
    if usl and lsl:
        out_spec = count_out_of_limits(data, usl, lsl)
        if out_spec > 0:
            st.warning(f"⚠️ **Tolerans Dışı:** {out_spec} adet ölçüm ({out_spec/len(data)*100:.1f}%) spesifikasyon limitlerinin dışında!")

# ============================================
# GRAFİKLER
# ============================================
st.markdown('<p class="section-header">📈 Grafikler</p>', unsafe_allow_html=True)

# Grafik 1: Kontrol Grafiği (Tam genişlik)
st.markdown("#### 📈 I-MR Kontrol Grafiği")
st.markdown("""
<div class="param-info">
Bu grafik ölçümlerin zaman içindeki değişimini gösterir. 
<b>Yeşil çizgi:</b> Ortalama | <b>Kırmızı kesikli:</b> Kontrol limitleri (±3σ) | <b>Turuncu:</b> Tolerans limitleri
</div>
""", unsafe_allow_html=True)

fig = go.Figure()

# X ekseni
if COL_DATE in df.columns:
    x_axis = df.loc[data.index, COL_DATE]
else:
    x_axis = list(range(len(data)))

# Ana veri
fig.add_trace(go.Scatter(
    x=x_axis, y=data,
    mode='lines+markers',
    name='Ölçüm Değeri',
    line=dict(color='#3498db', width=1.5),
    marker=dict(size=5, color='#3498db'),
    hovertemplate='<b>Değer:</b> %{y:.4f}<br><b>Tarih:</b> %{x}<extra></extra>'
))

# Sigma bantları
fig.add_hrect(y0=metrics['mean']-metrics['sigma_mr'], y1=metrics['mean']+metrics['sigma_mr'],
              fillcolor="rgba(46, 204, 113, 0.15)", line_width=0, 
              annotation_text="±1σ", annotation_position="right")
fig.add_hrect(y0=metrics['mean']-2*metrics['sigma_mr'], y1=metrics['mean']+2*metrics['sigma_mr'],
              fillcolor="rgba(241, 196, 15, 0.08)", line_width=0)

# Ortalama
fig.add_hline(y=metrics['mean'], line_color='#27ae60', line_width=2.5,
              annotation_text=f"Ortalama: {metrics['mean']:.4f}", 
              annotation_position="left",
              annotation_font_size=12)

# Kontrol limitleri
fig.add_hline(y=metrics['ucl'], line_dash='dash', line_color='#e74c3c', line_width=2,
              annotation_text=f"UCL: {metrics['ucl']:.4f}", 
              annotation_position="right",
              annotation_font_size=11)
fig.add_hline(y=metrics['lcl'], line_dash='dash', line_color='#e74c3c', line_width=2,
              annotation_text=f"LCL: {metrics['lcl']:.4f}", 
              annotation_position="right",
              annotation_font_size=11)

# Tolerans limitleri
if usl:
    fig.add_hline(y=usl, line_color='#f39c12', line_width=2.5, line_dash='dot',
                  annotation_text=f"USL: {usl}", annotation_position="right")
if lsl:
    fig.add_hline(y=lsl, line_color='#f39c12', line_width=2.5, line_dash='dot',
                  annotation_text=f"LSL: {lsl}", annotation_position="right")

# Kontrol dışı noktalar
out_mask = (data > metrics['ucl']) | (data < metrics['lcl'])
out_points = data[out_mask]
if len(out_points) > 0:
    if COL_DATE in df.columns:
        out_x = df.loc[out_points.index, COL_DATE]
    else:
        out_x = out_points.index.tolist()
    fig.add_trace(go.Scatter(
        x=out_x, y=out_points,
        mode='markers',
        name='Kontrol Dışı Noktalar',
        marker=dict(color='#e74c3c', size=12, symbol='x', line=dict(width=2))
    ))

fig.update_layout(
    height=550,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(l=60, r=120, t=60, b=60),
    xaxis_title="Tarih" if COL_DATE in df.columns else "Sıra No",
    yaxis_title=f"{secili_param} ({param_info['birim']})",
    font=dict(size=12)
)

st.plotly_chart(fig, use_container_width=True)

# Alt grafikler
st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

col_hist, col_mr = st.columns(2)

with col_hist:
    st.markdown("#### 📊 Histogram (Dağılım)")
    st.markdown("""
    <div class="param-info" style="font-size: 0.85rem;">
    Verilerin nasıl dağıldığını gösterir. Normal dağılıma yakınlık süreç stabilitesini gösterir.
    </div>
    """, unsafe_allow_html=True)
    
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Histogram(
        x=data,
        nbinsx=30,
        name='Frekans',
        marker_color='#3498db',
        opacity=0.75
    ))
    
    # Normal eğri
    x_range = np.linspace(data.min(), data.max(), 100)
    y_norm = stats.norm.pdf(x_range, metrics['mean'], metrics['sigma_mr'])
    scale = len(data) * (data.max() - data.min()) / 30
    
    fig_hist.add_trace(go.Scatter(
        x=x_range, y=y_norm * scale,
        mode='lines',
        name='Normal Dağılım',
        line=dict(color='#e74c3c', width=2.5)
    ))
    
    # Çizgiler
    fig_hist.add_vline(x=metrics['mean'], line_color='#27ae60', line_width=2)
    if usl: fig_hist.add_vline(x=usl, line_color='#f39c12', line_width=2, line_dash='dot')
    if lsl: fig_hist.add_vline(x=lsl, line_color='#f39c12', line_width=2, line_dash='dot')
    
    fig_hist.update_layout(
        height=400,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis_title=param_info['birim'],
        yaxis_title="Frekans"
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)

with col_mr:
    st.markdown("#### 📉 Moving Range (MR) Grafiği")
    st.markdown("""
    <div class="param-info" style="font-size: 0.85rem;">
    Ardışık ölçümler arasındaki farkı gösterir. Ani değişimleri tespit etmek için kullanılır.
    </div>
    """, unsafe_allow_html=True)
    
    mr = data.diff().abs()
    
    fig_mr = go.Figure()
    
    fig_mr.add_trace(go.Scatter(
        x=x_axis, y=mr,
        mode='lines+markers',
        name='Moving Range',
        line=dict(color='#9b59b6', width=1.5),
        marker=dict(size=4)
    ))
    
    fig_mr.add_hline(y=metrics['mr_mean'], line_color='#27ae60', line_width=2,
                      annotation_text=f"MR̄: {metrics['mr_mean']:.4f}")
    fig_mr.add_hline(y=metrics['mr_ucl'], line_dash='dash', line_color='#e74c3c', line_width=2,
                      annotation_text=f"UCL: {metrics['mr_ucl']:.4f}")
    
    fig_mr.update_layout(
        height=400,
        template="plotly_white",
        margin=dict(l=50, r=80, t=50, b=50),
        xaxis_title="Tarih" if COL_DATE in df.columns else "Sıra No",
        yaxis_title="Moving Range"
    )
    
    st.plotly_chart(fig_mr, use_container_width=True)

# ============================================
# DİRENÇ DETAY ANALİZİ
# ============================================
if 'DIRENC_FARKI' in df.columns and secili_param in ['Direnç Farkı (Başlangıç-Bitiş)', 'Başlangıç Direnç (Ω)', 'Bitiş Direnç (Ω)']:
    st.markdown('<p class="section-header">🔌 Direnç Detay Analizi</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="param-info">
    <h4>📐 Direnç Farkı Neden Önemli?</h4>
    <p>Üretim başlangıcı ve bitişi arasındaki direnç farkı, süreç stabilitesini gösterir. 
    İdeal durumda bu fark sıfıra yakın olmalıdır. Büyük farklar malzeme homojenliği veya 
    üretim parametrelerindeki değişimleri işaret edebilir.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    baslangic = df['KALİTE BAŞLANGIÇ ÖLÇÜLEN DİRENÇ'].dropna()
    bitis = df['KALİTE BİTİŞ ÖLÇÜLEN DİRENÇ'].dropna()
    fark = df['DIRENC_FARKI'].dropna()
    
    # Veri kontrolü
    if len(baslangic) == 0 or len(bitis) == 0 or len(fark) == 0:
        st.warning("⚠️ Direnç verisi yetersiz, analiz yapılamıyor.")
    else:
        with col_d1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">🔵 Başlangıç Direnç Ort.</div>
                <div class="metric-value">{baslangic.mean():.2f} Ω</div>
                <div class="metric-desc">Std: {baslangic.std():.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_d2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">🟢 Bitiş Direnç Ort.</div>
                <div class="metric-value">{bitis.mean():.2f} Ω</div>
                <div class="metric-desc">Std: {bitis.std():.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_d3:
            fark_ort = fark.mean()
            status = "status-good" if abs(fark_ort) < 0.5 else "status-warning" if abs(fark_ort) < 1 else "status-bad"
            st.markdown(f"""
            <div class="metric-box {status}">
                <div class="metric-label">📐 Ortalama Fark</div>
                <div class="metric-value">{fark_ort:.3f} Ω</div>
                <div class="metric-desc">Başlangıç - Bitiş</div>
            </div>
            """, unsafe_allow_html=True)
    
        # Direnç karşılaştırma grafiği
        st.markdown("#### 📊 Başlangıç vs Bitiş Direnç Karşılaştırması")
        
        fig_direnc = go.Figure()
        
        fig_direnc.add_trace(go.Box(
            y=baslangic,
            name='Başlangıç Direnç',
            marker_color='#3498db',
            boxpoints='outliers'
        ))
        
        fig_direnc.add_trace(go.Box(
            y=bitis,
            name='Bitiş Direnç',
            marker_color='#2ecc71',
            boxpoints='outliers'
        ))
        
        fig_direnc.add_trace(go.Box(
            y=fark,
            name='Fark (Baş-Bit)',
            marker_color='#9b59b6',
            boxpoints='outliers'
        ))
        
        fig_direnc.update_layout(
            height=450,
            template="plotly_white",
            yaxis_title="Direnç (Ω)",
            margin=dict(l=60, r=30, t=30, b=50)
        )
        
        st.plotly_chart(fig_direnc, use_container_width=True)

# ============================================
# MAKİNE KARŞILAŞTIRMA
# ============================================
if COL_MACHINE in df.columns and df[COL_MACHINE].nunique() > 1:
    st.markdown('<p class="section-header">🏭 Makine Bazlı Karşılaştırma</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="param-info">
    Farklı makinelerin performansını karşılaştırarak, hangi makinelerin daha stabil çalıştığını görebilirsiniz.
    </div>
    """, unsafe_allow_html=True)
    
    # Makine istatistikleri
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
        
        fig_box.add_hline(y=metrics['mean'], line_color='#27ae60', line_dash='dash',
                          annotation_text="Genel Ortalama")
        if usl:
            fig_box.add_hline(y=usl, line_color='#f39c12', annotation_text="USL")
        if lsl:
            fig_box.add_hline(y=lsl, line_color='#f39c12', annotation_text="LSL")
        
        fig_box.update_layout(
            height=450,
            template="plotly_white",
            showlegend=False,
            margin=dict(l=50, r=30, t=50, b=80)
        )
        
        st.plotly_chart(fig_box, use_container_width=True)

# ============================================
# PARAMETRE REHBERİ
# ============================================
st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

with st.expander("📚 SPC Parametreleri Rehberi - Tüm Açıklamalar"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📈 Temel İstatistikler
        
        | Parametre | Açıklama | Formül |
        |-----------|----------|--------|
        | **X̄ (Ortalama)** | Tüm değerlerin toplamının sayıya bölümü | Σx / n |
        | **σ (Sigma)** | Standart sapma - değişkenlik ölçüsü | √[Σ(x-X̄)²/(n-1)] |
        | **MR** | Moving Range - ardışık fark | ǀxᵢ - xᵢ₋₁ǀ |
        | **Range** | Maksimum - Minimum farkı | Max - Min |
        | **Medyan** | Ortanca değer | - |
        
        ### 🎯 Kontrol Limitleri
        
        | Limit | Açıklama | Hesaplama |
        |-------|----------|-----------|
        | **UCL** | Üst Kontrol Limiti | X̄ + 3σ |
        | **LCL** | Alt Kontrol Limiti | X̄ - 3σ |
        | **±1σ** | Değerlerin %68'i | X̄ ± σ |
        | **±2σ** | Değerlerin %95'i | X̄ ± 2σ |
        | **±3σ** | Değerlerin %99.7'si | X̄ ± 3σ |
        """)
    
    with col2:
        st.markdown("""
        ### 📏 Spesifikasyon Limitleri
        
        | Limit | Açıklama |
        |-------|----------|
        | **USL** | Üst Spesifikasyon Limiti - müşteri tarafından belirlenen maksimum |
        | **LSL** | Alt Spesifikasyon Limiti - müşteri tarafından belirlenen minimum |
        
        ### ⭐ Yeterlilik İndeksleri
        
        | İndeks | Açıklama | Formül | Hedef |
        |--------|----------|--------|-------|
        | **Cp** | Potansiyel yetenek | (USL-LSL) / 6σ | ≥ 1.33 |
        | **Cpk** | Gerçek yetenek | min(CPU, CPL) | ≥ 1.33 |
        | **Pp** | Uzun vadeli potansiyel | (USL-LSL) / 6s | ≥ 1.33 |
        | **Ppk** | Uzun vadeli yetenek | min(PPU, PPL) | ≥ 1.33 |
        | **PPM** | Milyonda hatalı parça | - | < 66 |
        
        ### 📊 Cpk Değerlendirmesi
        
        | Aralık | Durum | Aksiyon |
        |--------|-------|---------|
        | < 1.00 | ❌ Yetersiz | Acil iyileştirme |
        | 1.00 - 1.33 | ⚠️ Kabul Edilebilir | İyileştirme planla |
        | 1.33 - 1.67 | ✅ İyi | İzlemeye devam |
        | > 1.67 | 🌟 Mükemmel | Mükemmelliği sürdür |
        """)

# Footer
st.markdown("---")

# secili_kesit tanımlı mı kontrol et
kesit_bilgisi = ""
if COL_GROUP in df.columns:
    try:
        if secili_kesit != 'Tümü':
            kesit_bilgisi = f"Kesit: <b>{secili_kesit}</b> | "
    except NameError:
        pass

st.markdown(f"""
<center>
<small>
📊 <b>SPC Analiz Sistemi</b> | {SIRKET_ISMI} | 
Analiz Edilen: <b>{len(data):,}</b> kayıt | 
{kesit_bilgisi}Parametre: <b>{secili_param}</b>
</small>
</center>
""", unsafe_allow_html=True)
