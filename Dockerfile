# 1. Temel imaj olarak Python 3.9 kullan
FROM python:3.9-slim

# 2. Çalışma klasörünü ayarla
WORKDIR /app

# 3. Gereklilik dosyasını kopyala ve kütüphaneleri yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Klasördeki tüm dosyaları (kod, excel, logo) içeri kopyala
COPY . .

# 5. Streamlit'in kullandığı portu dışarı aç
EXPOSE 8501

# 6. Uygulamayı başlat
CMD ["streamlit", "run", "spc_advanced.py", "--server.port=8501", "--server.address=0.0.0.0"]