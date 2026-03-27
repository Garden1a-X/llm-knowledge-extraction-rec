#!/usr/bin/env python3
"""Calculate mean and std for Ours-Noisy-Graph results on Video Games dataset"""

import statistics

# 5 runs of final_test_metrics
results = [
    {
        "NDCG@5": 0.31040830623546983,
        "NDCG@10": 0.3550112997337473,
        "NDCG@20": 0.39089349713820287,
        "Recall@5": 0.4319824003911024,
        "Recall@10": 0.5698753361036422,
        "Recall@20": 0.7118357369836226,
        "Precision@5": 0.08639648007822047,
        "Precision@10": 0.05698753361036422,
        "Precision@20": 0.03559178684918113,
        "Hit@5": 0.4319824003911024,
        "Hit@10": 0.5698753361036422,
        "Hit@20": 0.7118357369836226
    },
    {
        "NDCG@5": 0.3013182929820404,
        "NDCG@10": 0.3463043251028273,
        "NDCG@20": 0.3830163226770731,
        "Recall@5": 0.42138352481055974,
        "Recall@10": 0.5607724272793938,
        "Recall@20": 0.7059496455634319,
        "Precision@5": 0.08427670496211195,
        "Precision@10": 0.05607724272793938,
        "Precision@20": 0.035297482278171595,
        "Hit@5": 0.42138352481055974,
        "Hit@10": 0.5607724272793938,
        "Hit@20": 0.7059496455634319
    },
    {
        "NDCG@5": 0.3051468790268088,
        "NDCG@10": 0.3496137787595817,
        "NDCG@20": 0.38596821756737065,
        "Recall@5": 0.4268296260083109,
        "Recall@10": 0.5642532388169151,
        "Recall@20": 0.7079247127841604,
        "Precision@5": 0.08536592520166218,
        "Precision@10": 0.05642532388169152,
        "Precision@20": 0.035396235639208025,
        "Hit@5": 0.4268296260083109,
        "Hit@10": 0.5642532388169151,
        "Hit@20": 0.7079247127841604
    },
    {
        "NDCG@5": 0.3044980463460616,
        "NDCG@10": 0.34920825074337364,
        "NDCG@20": 0.38616735420376247,
        "Recall@5": 0.42697628941579074,
        "Recall@10": 0.565279882669274,
        "Recall@20": 0.7113761916401857,
        "Precision@5": 0.08539525788315815,
        "Precision@10": 0.0565279882669274,
        "Precision@20": 0.03556880958200929,
        "Hit@5": 0.42697628941579074,
        "Hit@10": 0.565279882669274,
        "Hit@20": 0.7113761916401857
    },
    {
        "NDCG@5": 0.31050689420557875,
        "NDCG@10": 0.35482430348213306,
        "NDCG@20": 0.39144984366797025,
        "Recall@5": 0.4336152529943779,
        "Recall@10": 0.5708824248350036,
        "Recall@20": 0.7156880958200929,
        "Precision@5": 0.08672305059887558,
        "Precision@10": 0.057088242483500355,
        "Precision@20": 0.035784404791004636,
        "Hit@5": 0.4336152529943779,
        "Hit@10": 0.5708824248350036,
        "Hit@20": 0.7156880958200929
    }
]

# Calculate mean and std for each metric
metrics = list(results[0].keys())
stats = {}

print("="*80)
print("Ours-Noisy-Graph Results (Video Games Dataset)")
print("="*80)
print("\nResults for 5 trials:\n")

for metric in metrics:
    values = [r[metric] for r in results]
    mean = statistics.mean(values)
    std = statistics.stdev(values)  # Sample std
    stats[metric] = {"mean": mean, "std": std}
    print(f"{metric:15s}: {mean:.4f} ± {std:.4f}")

print("\n" + "="*80)
print("Key Findings:")
print("="*80)

# Compare with Ours-Full (cleaned graph)
ours_full_ndcg10 = 0.3619
ours_full_recall10 = 0.5766
noisy_ndcg10 = stats["NDCG@10"]["mean"]
noisy_recall10 = stats["Recall@10"]["mean"]

print(f"\nOurs-Full (Cleaned Graph):")
print(f"  NDCG@10:  {ours_full_ndcg10:.4f}")
print(f"  Recall@10: {ours_full_recall10:.4f}")

print(f"\nOurs-Noisy-Graph (with 'other' entities):")
print(f"  NDCG@10:  {noisy_ndcg10:.4f} ± {stats['NDCG@10']['std']:.4f}")
print(f"  Recall@10: {noisy_recall10:.4f} ± {stats['Recall@10']['std']:.4f}")

ndcg_diff = ((noisy_ndcg10 - ours_full_ndcg10) / ours_full_ndcg10) * 100
recall_diff = ((noisy_recall10 - ours_full_recall10) / ours_full_recall10) * 100

print(f"\nPerformance Impact:")
print(f"  NDCG@10:  {ndcg_diff:+.1f}%")
print(f"  Recall@10: {recall_diff:+.1f}%")

if ndcg_diff < 0:
    print(f"\n✅ Conclusion: Even minor noise ('other' entities) degrades performance by {abs(ndcg_diff):.1f}%")
    print("   This validates the need for knowledge graph post-processing/filtering.")
else:
    print(f"\n⚠️ Note: Noisy graph shows {ndcg_diff:+.1f}% change compared to cleaned graph.")

print("\n" + "="*80)
