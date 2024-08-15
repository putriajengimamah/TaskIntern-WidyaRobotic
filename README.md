# Task Intern - Widya Robotic - Putri Ajeng Imamah
## Vehicle Detection, Tracking and Counting Project

Proyek ini memanfaatkan YOLOv8 untuk mendeteksi dan menghitung kendaraan yang melintasi garis gerbang miring di sebuah video. Sistem ini dirancang untuk memproses frame video secara real-time, melacak setiap kendaraan dengan ID unik, dan menghitungnya saat melintasi gerbang. Hasilnya adalah video output di mana kendaraan diidentifikasi, dilacak, dan dihitung secara akurat.


## Fitur Utama:
Deteksi Kendaraan: Menggunakan model YOLOv8 untuk mendeteksi kendaraan seperti mobil, bus, dan lainnya.
Pelacakan Kendaraan: Setiap kendaraan yang terdeteksi dilacak di setiap frame dengan ID unik yang ditampilkan dalam kotak pembatas.
Penghitungan Kendaraan: Kendaraan dihitung saat mereka melewati garis gerbang miring yang telah ditentukan.
Video Output: Video hasil proses yang menunjukkan kendaraan terdeteksi, dilacak, dan dihitung, disimpan sebagai file output.


## Tutorial Menjalankan Proyek:
1. Clone Repository dari GitHub
Langkah pertama adalah meng-clone repository proyek ke komputer lokal dengan perintah berikut:
git clone https://github.com/putriajengimamah/TaskIntern-WidyaRobotic.git

2. Masuk ke Direktori Proyek
cd TaskIntern-WidyaRobotic

3. Setelah cloning selesai, masuk ke direktori proyek dengan perintah:
cd TaskIntern-WidyaRobotic

4. Buat dan Aktifkan Virtual Environment
Disarankan untuk menggunakan virtual environment agar paket-paket yang diinstal tidak bentrok dengan sistem lain. Gunakan perintah berikut untuk membuat dan mengaktifkan virtual environment:
python -m venv .venv
.venv\Scripts\activate

5. Instal Dependensi
Semua dependensi yang dibutuhkan sudah tercantum di requirements.txt. Jalankan perintah berikut untuk menginstalnya:
pip install -r requirements.txt

6. Jalankan Script untuk Deteksi dan Penghitungan Kendaraan
Setelah semuanya siap, jalankan script utama untuk memproses video dan menghasilkan video output:
python countvehicle.py

7. Periksa Video Output
Setelah script selesai dijalankan, video output akan disimpan di direktori proyek dengan nama vehicle_counting_output.avi. Buka video ini untuk melihat hasil deteksi dan penghitungan kendaraan.

8. Menghentikan Virtual Environment
Jika sudah selesai, nonaktifkan virtual environment dengan perintah:
deactivate


## Penjelasan Kode:
- Mengimpor Library yang Diperlukan:

from ultralytics import YOLO
import cv2
import numpy as np

ultralytics: Library yang mencakup model YOLOv8 untuk deteksi objek.
cv2: Library OpenCV untuk pemrosesan video dan manipulasi gambar.
numpy: Digunakan untuk operasi numerik, seperti menghitung jarak dan menangani array.

- Memuat Model YOLOv8:

model = YOLO('model/yolov8l.pt')

Memuat model YOLOv8 dari path yang sudah ditentukan. Pastikan path-nya sesuai.

- Inisialisasi Video Capture:

video_path = 'data/toll_gate.mp4'
cap = cv2.VideoCapture(video_path)

Membuka file video yang akan diproses. Ganti path dengan file video yang diinginkan.

- Mendefinisikan Parameter:

vehicle_count = 0
tracked_vehicles = {}
vehicle_id = 0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

vehicle_count: Menghitung jumlah kendaraan yang melintasi gerbang.
tracked_vehicles: Menyimpan informasi kendaraan yang dilacak.
vehicle_id: ID unik untuk setiap kendaraan.
frame_width, frame_height: Dimensi frame video.
fps: Frame per detik dari video.

- Mendefinisikan Garis Gerbang Miring:

gate_start_point = (0, frame_height // 2 - 100)
gate_end_point = (frame_width, frame_height // 2 + 100)
gate_line_color = (0, 255, 0)
gate_thickness = 2

gate_start_point, gate_end_point: Koordinat untuk garis gerbang miring.
gate_line_color: Warna garis gerbang dalam video output (hijau).
gate_thickness: Ketebalan garis gerbang.

- Fungsi Pelacakan dan Penghitungan Kendaraan:

is_crossing_gate: Menentukan apakah kendaraan melintasi gerbang berdasarkan posisi y.
euclidean_distance: Menghitung jarak antara dua titik untuk mencocokkan kendaraan.
get_class_name: Mengembalikan nama kelas kendaraan berdasarkan ID kelas.

- Memproses Frame Video:

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, show=False)

Loop utama untuk memproses setiap frame, mendeteksi, melacak, dan menghitung kendaraan.

- Menggambar pada Frame dan Menyimpan Output:

out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
Menggambar garis gerbang, bounding box, dan ID kendaraan pada setiap frame, lalu menyimpan hasilnya ke video output.

Dengan mengikuti langkah-langkah di atas, proyek ini bisa dijalankan dengan mudah. Selamat mencoba!
