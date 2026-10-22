## **1. Arsitektur & Cara Kerja (cGAN)**
Model ini menggunakan **Conditional GAN (cGAN)** karena dataset memiliki 2 kelas spesifik (*Storage Tanks* dan *Parking Lot*)[cite: 6]:
* **Generator ($G$):** Menerima input *random noise* $z$ serta *label class* (melalui *Embedding layer*) untuk dikonversi menjadi citra palsu (*fake image*) berukuran $28 \times 28$ piksel[cite: 6].
* **Discriminator ($D$):** Menerima masukan berupa citra (asli atau palsu) beserta label kelasnya, lalu bertugas membedakan apakah citra tersebut *Real* atau *Fake*[cite: 6].
* **Proses Training:** Dilakukan secara *adversarial* melalui *backpropagation* secara bergantian hingga Generator mampu menghasilkan citra yang cukup meyakinkan untuk menipu Discriminator[cite: 6].

---

## **2. Dataset & Preprocessing**
* **Total Dataset:** 16.016 gambar[cite: 6].
* **Kelas Target:** 
  * *Parking Lot* (8.007 gambar)[cite: 6]
  * *Storage Tanks* (8.009 gambar)[cite: 6]
* **Langkah Preprocessing:**
  1. Konversi citra ke *grayscale* ('L') dan *resize* ke ukuran $28 \times 28$ piksel[cite: 6].
  2. Normalisasi nilai piksel ke rentang $[-1, 1]$ (sesuai aktivasi `Tanh` pada Generator)[cite: 6].
  3. Pembagian data (*Train*, *Validation*, *Test* dengan *stratified split*)[cite: 6].

---

## **3. Struktur Model & Hiperparameter**
* **Latent Dimension:** 100[cite: 6]
* **Batch Size:** 64[cite: 6]
* **Epochs:** 100[cite: 6]
* **Optimizer:** Adam ($\text{lr} = 2 \times 10^{-4}, \beta_1 = 0.5, \beta_2 = 0.999$)[cite: 6]
* **Loss Function:** Binary Cross-Entropy Loss (`BCELoss`)[cite: 6]

---

## **4. Cara Menjalankan Kode**
Pastikan pustaka berikut sudah terinstal di lingkungan Python Anda:
```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow scipy
