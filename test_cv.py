import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from modules.cross_validation import run_cross_validation

small_data = [
    {'normalized': 'energi terbarukan bagus', 'label': 0},
    {'normalized': 'hoax berita palsu energi', 'label': 1},
    {'normalized': 'solar panel murah efisien', 'label': 0},
    {'normalized': 'bohong pemerintah energi krisis', 'label': 1},
    {'normalized': 'listrik tenaga surya bersih', 'label': 0},
    {'normalized': 'penipuan data energi palsu', 'label': 1},
    {'normalized': 'pembangkit angin ramah lingkungan', 'label': 0},
    {'normalized': 'konspirasi minyak energi mahal', 'label': 1},
    {'normalized': 'nuklir aman modern teknologi', 'label': 0},
    {'normalized': 'hoax bbm naik bohong rakyat', 'label': 1},
] * 3  # 30 samples

print("=== RUN 1 ===")
r1 = run_cross_validation(small_data, k=3, seed=42, ensemble_method='hard')
rf1 = [r1['fold_results'][i]['evaluations']['Random Forest']['accuracy'] for i in range(3)]
en1 = [r1['fold_results'][i]['evaluations']['Ensemble (Voting)']['accuracy'] for i in range(3)]
print(f"RF accuracies:       {rf1}")
print(f"Ensemble accuracies: {en1}")

print("\n=== RUN 2 (harus IDENTIK dengan RUN 1) ===")
r2 = run_cross_validation(small_data, k=3, seed=42, ensemble_method='hard')
rf2 = [r2['fold_results'][i]['evaluations']['Random Forest']['accuracy'] for i in range(3)]
en2 = [r2['fold_results'][i]['evaluations']['Ensemble (Voting)']['accuracy'] for i in range(3)]
print(f"RF accuracies:       {rf2}")
print(f"Ensemble accuracies: {en2}")

print("\n=== RUN 3 (harus IDENTIK lagi) ===")
r3 = run_cross_validation(small_data, k=3, seed=42, ensemble_method='hard')
rf3 = [r3['fold_results'][i]['evaluations']['Random Forest']['accuracy'] for i in range(3)]
en3 = [r3['fold_results'][i]['evaluations']['Ensemble (Voting)']['accuracy'] for i in range(3)]
print(f"RF accuracies:       {rf3}")
print(f"Ensemble accuracies: {en3}")

print(f"\nKonsisten? RF={rf1==rf2==rf3}, Ensemble={en1==en2==en3}")
