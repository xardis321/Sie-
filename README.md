ZBIÓR KODÓW ZWIĄZANYCH Z SIECIAMI



$SIEC.PY
Program jest implementacją prostej sztucznej sieci neuronowej 
Program korzysta z funkcji aktywacji, funkcji straty i metody trenowania dla obu typów problemów (regresji i klasyfikacji).
Funkcja "darken_moody" do dostosowania współczynnika uczenia w trakcie procesu uczenia.
Sieć ma strukturę konfigurowalną za pomocą parametrów, takich jak liczba warstw, liczba jednostek w warstwach, funkcje aktywacji, itp.
Program można wykorzystać do eksperymentów z różnymi strukturami sieci i problemami uczenia maszynowego.




#regresja
Program zawiera funkcje do generowania danych do zadania regresji, wczytywania tych danych, oraz implementacje różnych funkcji aktywacji (linearnej, sigmoid, tanh, ReLU) i obliczania ich pochodnych. 
Program demonstruje także wykorzystanie tych funkcji aktywacji, obliczając ich wyjścia dla konkretnej wartości wejściowej.


#definicja
Kod definiuje klasy reprezentujące sieć neuronową, funkcje aktywacji i funkcje straty. Zapewnia funkcje do generowania danych do zadań regresji i klasyfikacji.
Testuje działanie sieci na przykładowych danych. Ostateczne zastosowanie obejmuje trenowanie sieci w zadaniach regresji sinusoidalnej z szumem oraz klasyfikacji dwóch klas na podstawie losowych danych punktów.


#implementacja
zawiera implementacje modeli sieci neuronowych w TensorFlow do rozwiązywania zadań regresji i klasyfikacji. Dla zadań regresji używany model z jednym i dwoma ukrytymi warstwami, a także z warstwą BatchNormalization.



#klasyfikacja
Kod ten korzysta z TensorFlow i TensorFlow Datasets do trenowania modeli klasyfikacyjnych na zbiorze danych zdjęć kwiatów.Kod ten służy eksperymentalnemu zrozumieniu wpływu różnych architektur modeli i technik augmentacji danych na jakość klasyfikacji zdjęć kwiatów.
