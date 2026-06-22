Replicare Anomaly Transformer: Detectarea Anomaliilor în Serii de Timp

Descriere: 
Acest proiect reprezintă replicarea și analiza critică a articolului științific "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy" (ICLR 2022). Scopul principal este implementarea arhitecturii propuse pentru detectarea nesupervizată a anomaliilor în serii de timp, utilizând mecanismul dual Anomaly-Attention și o strategie de optimizare Minimax.

Arhitectura Modelului:
Modelul se distinge de autoencoderele clasice prin modelarea simultană a două tipuri de asocieri temporale:

Prior-Association: Modelează tendința anomaliilor de a se concentra în regiuni adiacente, folosind un kernel Gaussian cu un parametru de scală ce poate fi învățat.

Series-Association: Învață asocierile reale din datele brute folosind mecanismul standard de self-attention.

Diferența dintre aceste două distribuții (calculată prin divergența Kullback-Leibler simetrizată pe mai multe straturi) formează Association Discrepancy, o metrică robustă care scoate în evidență comportamentul anormal.

Modelul a fost adaptat și validat pe următoarele seturi de date de referință (benchmark-uri):
SMD (Server Machine Dataset) - 38 dimensiuni

Cerințe și Instalare:
Proiectul este scris în Python și folosește ecosistemul PyTorch pentru arhitectura de Deep Learning.

1. Creați un mediu virtual (opțional, dar recomandat):
		conda create -n anomaly_transformer python=3.9
		conda activate anomaly_transformer

2. Instalați dependențele necesare rulând următoarea comandă în rădăcina proiectului:
		pip install -r requirements.txt

3. Rulati testare_model.py, contine deja modelul antrenat din notebook
		python test_model.py

