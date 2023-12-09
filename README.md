Terdapat 2 program GA, yang file bernama GA_PCOS.ipynb itu dibuat dengan menggunakan colab sedangkan yang GeneticAlgorithm.py itu menggunakan VSCode. kedua program memiliki inti yang sama tapi memiliki struktur program yang berbeda. Saya sarankan yang GeneticAlgorithm.py

Dalam program ini, Algoritma Genetika (GA) digunakan untuk pemilihan fitur guna mengoptimalkan kinerja RandomForestClassifier untuk memprediksi label PCOS (Sindrom Ovarium Polikistik). 

Teknik seleksi yang digunakan adalah seleksi elitisme. Seleksi ini memilih sejumlah individu terbaik (berdasarkan skor kebugaran) dari generasi saat ini dan mentransfernya ke generasi berikutnya tanpa mengalami perubahan atau crossover.

Teknik crossover yang digunakan adalah one-point crossover. Pada one-point crossover, suatu titik acak dipilih di dalam kromosom, dan nilai-nilai di antara titik tersebut dipertukarkan antara dua kromosom untuk menghasilkan dua anak baru.

teknik klasifikasi menggunakan Random Forest Clasifier

