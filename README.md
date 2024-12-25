
<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTb0qu3fC1MDco5o3odZ7t4L6BGu1gXC66EjIO4MngVXcFbpn4i2qiitFgLNVxQ5dyp8I&usqp=CAU" alt="Logo" />
</p>

# Klasifikasi Bahasa Isyarat Amerika (ASL) Menggunakan CNN  

## 📖 Deskripsi Proyek  
Proyek ini mengimplementasikan model Convolutional Neural Network (CNN) untuk mengklasifikasikan Bahasa Isyarat Amerika (ASL). Pengguna dapat memilih salah satu dari tiga arsitektur populer, yaitu **VGG19**, **ResNet101 v3**, atau **InceptionV3**, sesuai dengan kebutuhan mereka. Proyek ini dirancang untuk mendukung komunikasi yang lebih inklusif bagi penyandang disabilitas pendengaran.  

## 📂 Dataset & Models
Klik Icon dibawah:

[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1Hmd8IblC45SEiE9vvQSAwE9hio62F6V0?usp=drive_link)
[![Kaggle](https://img.shields.io/badge/Kaggle-ASL%20Alphabet-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  

## 🛠️ Fitur Utama  
- **Pemilihan Model:**  
  - **VGG19:** 
  - **ResNet101 v3:**  
  - **InceptionV3:** 
- **Klasifikasi Real-Time:**  
  Pengguna dapat mengunggah gambar isyarat tangan melalui antarmuka web berbasis Flask untuk mendapatkan hasil klasifikasi secara instan.  
- **Fleksibilitas:**  
  Memungkinkan pengguna memilih model yang paling sesuai dengan kebutuhan performa dan akurasi.  

## ⚙️ Instalasi  
Ikuti langkah-langkah berikut untuk menjalankan proyek ini:
1. Pastikan sudah download 📂 Dataset & Models

2. **Clone repository ini:**  
   ```bash
   git clone https://github.com/Wildanamru/Implementasi-CNN-VGG19-untuk-Klasifikasi-Bahasa-Isyarat-Amerika-ASL-Berbasis-Flask.git
   ```
2. **Buat environment Python virtual & aktifkan:**  
   ```bash
   python -m venv venv
   venv/Scripts/activate #Windows
   ```
3. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt  
   ```
4. **Jalankan aplikasi web Flask:**  
   ```bash
   python app.py  
   ```
5. **Akses aplikasi melalui browser di alamat berikut:**  
   ```bash
   http://127.0.0.1:5000   
   ```
## 📊 Evaluasi Model
Model dievaluasi menggunakan metrik berikut:


  📦VGG19
      Akurasi: 100%

    
  📦ResNet101 v3
      Akurasi: 100%

    
  📦InceptionV3
      Akurasi:96%
    
## 👩‍💻 Environment Variables

Paket Library yang digunakan untuk deploy flask:

`Python==3.10.14`
`cv2`
`tensorflow`
`flask`
`mediapipe`

## 📧 Author
Wildan Amru Hidayat (202110370311280)

