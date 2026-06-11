import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 1. PROSES MEMBACA DATASET DENGAN AUTO-DETEKSI SEPARATOR
file_name = 'Dataset Rusfaldi Andika.csv'
separators = [';', ',']
df = None

for sep in separators:
    try:
        temp_df = pd.read_csv(file_name, sep=sep)
        cols = [str(c).upper() for c in temp_df.columns]
        # Cek apakah kolom-kolom kunci ada di separator ini
        has_x1 = any('UMP' in c or 'X1' in c for c in cols)
        has_x2 = any('TPT' in c or 'X2' in c for c in cols)
        has_y = any('MISKIN' in c or 'Y' in c for c in cols)
        
        if has_x1 and has_x2 and has_y:
            df = temp_df
            break
    except:
        continue

if df is None:
    # Jika gagal auto-deteksi, load default standar
    try:
        df = pd.read_csv(file_name, sep=';')
    except FileNotFoundError:
        print(f"❌ ERROR: File '{file_name}' tidak ditemukan di folder ini!")
        exit()

print("✅ Dataset berhasil dimuat!")

# 2. AUTO-MAPPING NAMA KOLOM (Mencegah error spasi atau typo header)
mapping = {}
for c in df.columns:
    c_upper = str(c).upper()
    if 'UMP' in c_upper or 'X1' in c_upper:
        mapping['UMP(X1)'] = c
    elif 'TPT' in c_upper or 'X2' in c_upper:
        mapping['TPT(X2)'] = c
    elif 'MISKIN' in c_upper or 'Y' in c_upper:
        mapping['%Miskin(Y)'] = c

df = df.rename(columns={v: k for k, v in mapping.items()})

# 3. PROSES PEMBERSIHAN ELEMEN MENGGUNAKAN REGEX (ANTI GAGAL)
def clean_val(val, is_ump=False):
    if pd.isna(val): return val
    s = str(val).strip()
    s = re.sub(r'[^\d.,-]', '', s) # Hapus persen %, lambang Rp, dll.
    if not s: return None
    
    if is_ump:
        if '.' in s and ',' in s:
            s = s.replace('.', '').replace(',', '.')
        elif ',' in s:
            s = s.replace(',', '.')
        elif '.' in s:
            if s.count('.') > 1 or len(s.split('.')[-1]) == 3:
                s = s.replace('.', '')
    else:
        if ',' in s:
            s = s.replace(',', '.')
    try:
        return float(s)
    except:
        return None

df['UMP(X1)'] = df['UMP(X1)'].apply(lambda x: clean_val(x, is_ump=True))
df['TPT(X2)'] = df['TPT(X2)'].apply(lambda x: clean_val(x, is_ump=False))
df['%Miskin(Y)'] = df['%Miskin(Y)'].apply(lambda x: clean_val(x, is_ump=False))

# Buang baris kosong
df_clean = df.dropna(subset=['UMP(X1)', 'TPT(X2)', '%Miskin(Y)'])

if len(df_clean) == 0:
    print("❌ ERROR: Data menjadi 0 baris setelah dibersihkan. Sila cek struktur isi file lu!")
    exit()

# 4. PROSES METODE REGRESI LINEAR BERGANDA
X = df_clean[['UMP(X1)', 'TPT(X2)']]
Y = df_clean['%Miskin(Y)']
X_dengan_konstanta = sm.add_constant(X)

model = sm.OLS(Y, X_dengan_konstanta).fit()

# Menampilkan Ringkasan Hasil Statistik untuk UAS Lu
print("\n" + "="*65)
print("             HASIL ANALISIS REGRESI LINEAR BERGANDA")
print("="*65)
print(model.summary())
print("="*65)

# 5. MEMBUAT VISUALISASI GRAFIK
df_clean['Prediksi_Miskin'] = model.predict(X_dengan_konstanta)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='%Miskin(Y)', y='Prediksi_Miskin', data=df_clean, color='blue', label='Data Aktual')
plt.plot([df_clean['%Miskin(Y)'].min(), df_clean['%Miskin(Y)'].max()], 
         [df_clean['%Miskin(Y)'].min(), df_clean['%Miskin(Y)'].max()], 
         color='red', linestyle='--', linewidth=2, label='Garis Regresi')

plt.xlabel('Aktual Persentase Penduduk Miskin (%)')
plt.ylabel('Hasil Prediksi Model (%)')
plt.title('Grafik Analisis Regresi Linear Berganda\nRusfaldi Andika - Dataset Kemiskinan')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Menyimpan hasil grafik jadi gambar
plt.savefig('grafik_hasil_regresi.png')
print("\n✅ [SUKSES] Grafik berhasil disimpan sebagai 'grafik_hasil_regresi.png'")
plt.show()