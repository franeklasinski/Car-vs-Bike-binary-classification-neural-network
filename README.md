# Car vs Bike Classification Project

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

## Opis Projektu

Projekt klasyfikacji obrazów wykorzystujący głębokie sieci neuronowe do rozróżniania między samochodami a rowerami. Model został zbudowany w PyTorch z wykorzystaniem zaawansowanych technik deep learningu.

## Cele Projektu

- **Klasyfikacja binarna**: Rozróżnianie między obrazami samochodów i rowerów
- **Wysoka dokładność**: Osiągnięcie accuracy > 95% na zbiorze walidacyjnym
- **Analiza wydajności**: Szczegółowa ewaluacja modelu z metrykami i wizualizacjami

## Dataset
**kaggle**: https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset

## Struktura Projektu

```
car-vs-bike/
├── car-bike.ipynb          # Główny notebook z modelem
├── Car-Bike-Dataset/       # Zbiór danych
│   ├── Car/                # Obrazy samochodów
│   └── Bike/               # Obrazy rowerów
├── best_car_bike_model.pth # Najlepszy wytrenowany model
├── runs/                   # Logi TensorBoard
└── README.md               # Ten plik
```

## Funkcjonalności

### Przetwarzanie Danych
- **Data Augmentation**: Rotacja, odbicia, zmiana kolorów
- **Normalizacja**: Standardowa normalizacja ImageNet
- **Podział danych**: 80% trening / 20% walidacja

### Architektura Modelu
- **Typ**: Convolutional Neural Network (CNN)
- **Warstwy**: 4 bloki konwolucyjne + 3-warstwowy klasyfikator
- **Parametry**: ~11M parametrów
- **Regularyzacja**: BatchNorm, Dropout, Weight Decay

```python
# Architektura modelu
- Block 1: Conv2d(3→64) + BatchNorm + ReLU + Conv2d(64→64) + MaxPool + Dropout
- Block 2: Conv2d(64→128) + BatchNorm + ReLU + Conv2d(128→128) + MaxPool + Dropout
- Block 3: Conv2d(128→256) + BatchNorm + ReLU + Conv2d(256→256) + MaxPool + Dropout
- Block 4: Conv2d(256→512) + BatchNorm + ReLU + AdaptiveAvgPool2d
- Classifier: Linear(8192→1024) + ReLU + Dropout + Linear(1024→256) + Linear(256→2)
```

### Trening i Optymalizacja
- **Optimizer**: AdamW z weight decay
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Zapobiega overfittingowi
- **Loss Function**: CrossEntropyLoss

### Ewaluacja i Wizualizacje
- **Confusion Matrix**: Macierz pomyłek
- **Classification Report**: Precision, Recall, F1-Score
- **Wykresy treningu**: Loss i accuracy curves
- **Przykładowe predykcje**: Obrazy z pewnością predykcji

## Wyniki

### Metryki Wydajności
- **Accuracy**: ~96%
- **Precision**: Bike: 0.94, Car: 0.96
- **Recall**: Bike: 0.96, Car: 0.94
- **F1-Score**: Bike: 0.95, Car: 0.95

### Analiza Treningu
- **Epochs**: 30 (z early stopping)
- **Best Validation Accuracy**: ~96%
- **Overfitting Gap**: < 2%
- **Model Stability**: Wysoka (std < 0.01)

## Uruchomienie

1. **Klonowanie repozytorium**:
```bash
git clone https://github.com/franeklasinski/Car-vs-Bike-binary-classification-neural-network.git
cd Car-vs-Bike-binary-classification-neural-network
```

2. **Instalacja zależności**:
```bash
pip install torch torchvision matplotlib numpy pandas seaborn scikit-learn
```

3. **Uruchomienie notebooka**:
```bash
jupyter notebook car-bike.ipynb
```

**Autor**: Franciszek Lasiński
