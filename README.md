# Projekt inżynierski - Automatyka i Robotyka
 
## Temat: Lokalizacja i zliczanie obiektów z wykorzystaniem fuzji danych z czujników bezzałogowego statku powietrznego

<p align="center">
  <img src="media/rosbag_20000101_024312_median_gif.gif" alt="Schemat Drona" width="45%">
  <img src="media/drone_photo.jpeg" alt="Zdjęcie Drona" width="35%">
</p>

## Opis Projektu
Celem pracy dyplomowej jest opracowanie, implementacja oraz ewaluacja systemu lokalizacji obiektów z wykorzystaniem fuzji danych pozyskanych z czujników zamontowanych na pokładzie bezzałogowego statku powietrznego (drona). Użyte czujniki to m.in. kamera RGB, kamera głębi oraz IMU.

---

## Struktura Pracy

### 🛠️ **Realizacja**
1. Montaż kamery głębi Intel RealSense D435i, modułu GPS RTK oraz jednostki obliczeniowej Jetson Xavier NX na dronie rozwijanym w ramach Studenckiego Koła Naukowego AGH AVADER.
2. Przygotowanie makiety sadu z piłkami imitującymi owoce, a następnie testy z użyciem prawdziwych owoców.
3. Przeprowadzenie nalotów z wykorzystaniem zdalnego sterowania (RC) i zapis danych w formacie ROS (rosbag).
4. Uruchomienie systemu precyzyjnej lokalizacji GPS RTK na pokładzie drona.

### 📊 **Analiza i Przetwarzanie Danych**
- Opracowanie systemu lokalizacji obiektów z wykorzystaniem fuzji danych z czujników.
- Wytrenowanie oraz analiza wyników 16 modeli YOLO w wersjach v11 oraz v8, z różnymi stopniami kwantyzacji.
- Implementacja i porównanie wydajności algorytmów:
  - **Single-object tracking** (OpenCV): MedianFlow, CSRT, KCF.
  - **Multi-object tracking**: ByteTrack, Bot-SORT.
  - Algorytm sortujący SORT połączony z SOT.
- Ewaluacja skuteczności i złożoności obliczeniowej zaprojektowanych algorytmów.

### 🛡️ **Implementacja Wbudowana**
- Implementacja algorytmów na platformie Jetson Xavier NX w celu zliczania obiektów w czasie rzeczywistym na pokładzie drona.
- Testy systemu w trakcie lotów drona w czasie rzeczywistym.

---

**Temat pracy:**  
🇵🇱 Lokalizacja i zliczanie obiektów z wykorzystaniem fuzji danych z czujników bezzałogowego statku powietrznego  
🇬🇧 Locating and counting objects using data fusion from unmanned aerial vehicle sensors

**Autor:** Remigiusz Mietła  
**Promotor:** dr inż. Tomasz Kryjak  
**Kierunek:** Automatyka i Robotyka  
**Wydział:** WEAIiIB  
**Uczelnia:** AGH - Akademia Górniczo-Hutnicza im. Stanisława Staszica w Krakowie
