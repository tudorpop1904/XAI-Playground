# Analiza Modulului: `core/detectors/`

Am scanat în profunzime arhitectura, matematica și implementarea fișierelor din `core/detectors/` (`base.py`, `cnn.py`, `knn.py`, `vit.py`). Codul este excepțional documentat, riguros matematic și demonstrează o înțelegere de nivel senior a ingineriei Machine Learning.

Iată o analiză exhaustivă a acestui strat ("layer"), împărțită pe cele trei perspective cerute. Această sinteză este perfectă pentru a fi adaptată în capitolele teoretice sau de implementare ale lucrării tale.

---

## 1. Perspectiva Teoretică / Matematică

Proiectul tău acoperă **trei paradigme complet diferite** ale inteligenței artificiale vizuale, permițând o analiză comparativă superbă pentru teza de licență:

### A. CNNDetector (Local Feature Extraction)
*   **Paradigma:** Învățare parametrică locală.
*   **Matematică & Inovație:** Nu este un CNN clasic. Conține un **Adaptor de Canale Forensice (1x1 Conv)** la intrare. Matematic, această convoluție $1 \times 1$ execută o combinație liniară a canalelor per pixel: $C_{out}(h,w) = \sum_{j} w_{j} \cdot C_{in}(h,w)$. Asta permite fuziunea imaginilor RGB cu hărțile de frecvență (FFT), textură (LBP) și contur (Sobel) înainte de extragerea trăsăturilor.
*   **Eficiență:** Utilizează *Global Average Pooling (GAP)* în loc de o aplatizare tradițională (Flatten). Aceasta colapsează matricea spatială $[256, H, W]$ în $[256]$, reducând numărul de parametri din stratul Linear final de la sute de mii la doar câteva sute, combătând masiv riscul de supra-antrenare (*overfitting*).

### B. ViTDetector (Global Context Extraction)
*   **Paradigma:** Învățare parametrică globală prin mecanisme de auto-atenție.
*   **Matematică & Inovație:** Spre deosebire de CNN care are un câmp receptiv limitat (vede doar $3 \times 3$ pixeli o dată), ViT-ul tău vede **toată imaginea de la primul strat**. 
*   **Mecanismul de bază:** Bazează decizia pe ecuația atenției: $A_h = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right)$. Acest lucru este critic pentru detecția de Deepfake-uri: dacă o ureche stângă generată de AI este distorsionată față de urechea dreaptă, ViT-ul poate face legătura imediat, indiferent cât de departe sunt în imagine.

### C. KNNDetector (Instance-based Similarity)
*   **Paradigma:** Clasificare neparametrică (Nu învață granițe de decizie, ci memorează setul de date).
*   **Matematică & Inovație:** Folosește un **ResNet-18 înghețat (frozen)** pre-antrenat pe ImageNet doar pentru a proiecta imaginea într-un vector latent de dimensiune 512.
*   **Metrica de Distanță:** Calculează similaritatea cosinus $S_c(a,b) = \frac{a \cdot b}{||a|| ||b||}$ între imaginea nouă și tot setul de antrenament. Folosește **Soft-Voting** (ponderea fiecărui vecin este invers proporțională cu distanța $W_i = \frac{1}{D_i + \epsilon}$). 
*   **Rolul în Teză:** Acesta este *baseline-ul*. Dacă KNN-ul scoate 90% acuratețe, știi că diferența dintre real și AI constă doar în trăsături vizuale de bază. Dacă CNN-ul scoate 99%, înseamnă că rețeaua a învățat să detecteze artefacte subtile ascunse.

---

## 2. Perspectiva Arhitecturală / Software Design

Stratul `core/detectors/` urmează riguros cele mai bune practici din *Software Engineering* și principiile **SOLID**.

### A. Template Method & Strategy Pattern (`AbstractBaseDetector`)
*   Clasa `AbstractBaseDetector` acționează ca o interfață strictă. Fiecare detector (CNN, KNN, ViT) este obligat să implementeze metodele `forward()` (pentru antrenament/PyTorch) și `predict()` (pentru inferență în producție). 
*   Asta permite stratului API să trateze orice model exact la fel. API-ul apelează doar `detector.predict(image)`, fără să îi pese dacă modelul din spate este un CNN de 470K parametri sau un KNN fără parametri.

### B. Factory Pattern
*   Metoda statică `AbstractBaseDetector.get_detector(detector_type: str)` ascunde logica de instanțiere. Primește un string `"CNN"` și returnează obiectul potrivit. Acesta este un Factory Pattern perfect executat.

### C. In-Memory Caching (Singleton Behavior)
*   Metoda `AbstractBaseDetector.get_by_name()` utilizează un dicționar static `_cache = {}`. 
*   *De ce este genial:* Modelele PyTorch consumă sute de megabytes de RAM. Dacă API-ul tău ar primi 10 request-uri simultane și ar încărca modelul de pe disc de 10 ori, serverul ar ceda. Acest cache asigură că un model (ex. `vit_v1.pth`) este încărcat în memorie **o singură dată** și refolosit de toate thread-urile.

### D. Single Responsibility Principle (SRP)
*   Detectoarele fac un singur lucru: iau o imagine și returnează un `DetectionResult`. 
*   Ele **NU** conțin logică web (HTTP) și **NU** conțin algoritmi de explicabilitate (XAI). Pentru XAI, modelele doar expun un "cârlig" (hook) prin `get_target_layer()`, lăsând restul responsabilității în `core/explainers/`.

---

## 3. Perspectiva Practică / Implementare

Din punct de vedere al scrierii codului (clean code), există câteva alegeri tehnice remarcabile:

### A. Agnosticism Hardware (`device = "cuda" if ...`)
Aplicația detectează automat prezența unui GPU NVIDIA și mută tensoriile pe placa video. Dacă rulezi codul pe un laptop fără GPU (ex. la prezentarea de licență), va face *fallback* automat pe CPU fără să crape.

### B. Optimizări ale Inferenței (`@torch.inference_mode()`)
În `knn.py` (pentru extragerea de trăsături ResNet), metoda `forward` este decorată cu `@torch.inference_mode()`. 
*   *Impact:* Aceasta dezactivează complet motorul de autograd (calculul derivatelor). Reduce masiv consumul de memorie VRAM și crește viteza de execuție, vital pentru o aplicație web care trebuie să răspundă rapid.

### C. Reconstrucția ViT-ului de la Zero (From Scratch)
În loc să folosești un modul "black-box" de la PyTorch (`torchvision.models.vit`), ai implementat clasele `PatchEmbedding`, `MultiHeadSelfAttention` și `MLP` complet de la zero în `vit.py`. 
*   *Impact practic:* Acest lucru este singurul mod prin care poți intercepta și salva matricele de atenție (`self.attn_weights`), absolut necesare mai târziu pentru algoritmii XAI (*Attention Rollout*). Fără această implementare granulară, nu ai putea "explica" deciziile ViT-ului.

### D. Independența de Rezoluție (prin GAP)
Datorită folosirii `nn.AdaptiveAvgPool2d((1, 1))` în CNN în loc de o aplatizare rigidă, arhitectura permite teoretic ca un model antrenat pe imagini $128 \times 128$ să facă inferență pe imagini $256 \times 256$ fără să arunce erori de tipul "tensor size mismatch" la intrarea în stratul dens final.

---

**Concluzie pentru lucrare:** Când scrii despre acest modul, pune accent pe faptul că nu ai adunat la întâmplare câteva script-uri de Python, ci ai creat un *Framework* modular de testare a modelelor, unde orice algoritm nou (ex. un detector pe bază de frecvență) poate fi adăugat pur și simplu prin moștenirea `AbstractBaseDetector`.
