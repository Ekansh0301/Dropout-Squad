"""
# Add this cell to your Kaggle notebook after the evaluation section
# This will help diagnose the calibration issue
"""

print("="*70)
print("DETAILED DIAGNOSTIC ANALYSIS")
print("="*70)

# Get predictions on validation set
predictions = trainer.predict(val_dataset)
pred_scores = 1 / (1 + np.exp(-predictions.predictions.squeeze()))
true_scores = predictions.label_ids

# 1. Score Distribution Analysis
print("\n📊 SCORE DISTRIBUTIONS")
print("-" * 70)
print("True Scores:")
print(f"  Range: [{true_scores.min():.3f}, {true_scores.max():.3f}]")
print(f"  Mean: {true_scores.mean():.3f} ± {true_scores.std():.3f}")
print(f"  Median: {np.median(true_scores):.3f}")

print("\nPredicted Scores:")
print(f"  Range: [{pred_scores.min():.3f}, {pred_scores.max():.3f}]")
print(f"  Mean: {pred_scores.mean():.3f} ± {pred_scores.std():.3f}")
print(f"  Median: {np.median(pred_scores):.3f}")

# 2. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Scatter plot
axes[0, 0].scatter(true_scores, pred_scores, alpha=0.3, s=20)
axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
# Add regression line
z = np.polyfit(true_scores, pred_scores, 1)
p = np.poly1d(z)
axes[0, 0].plot([0, 1], p([0, 1]), "b-", alpha=0.8, label=f'Actual fit: y={z[0]:.2f}x+{z[1]:.2f}')
axes[0, 0].set_xlabel('True Score', fontsize=12)
axes[0, 0].set_ylabel('Predicted Score', fontsize=12)
axes[0, 0].set_title('Predicted vs True Scores', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Distribution comparison
axes[0, 1].hist(true_scores, bins=50, alpha=0.6, label='True', color='blue', edgecolor='black')
axes[0, 1].hist(pred_scores, bins=50, alpha=0.6, label='Predicted', color='red', edgecolor='black')
axes[0, 1].set_xlabel('Score', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Score Distribution Comparison', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Error by true score
errors = np.abs(pred_scores - true_scores)
axes[1, 0].scatter(true_scores, errors, alpha=0.3, s=20)
axes[1, 0].axhline(y=errors.mean(), color='r', linestyle='--', 
                   label=f'Mean Error: {errors.mean():.3f}')
axes[1, 0].set_xlabel('True Score', fontsize=12)
axes[1, 0].set_ylabel('Absolute Error', fontsize=12)
axes[1, 0].set_title('Error vs True Score', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Residual plot
residuals = pred_scores - true_scores
axes[1, 1].scatter(true_scores, residuals, alpha=0.3, s=20)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].axhline(y=residuals.mean(), color='g', linestyle='--', 
                   label=f'Mean Residual: {residuals.mean():.3f}')
axes[1, 1].set_xlabel('True Score', fontsize=12)
axes[1, 1].set_ylabel('Residual (Pred - True)', fontsize=12)
axes[1, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostic_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Performance by Type
print("\n📊 PERFORMANCE BY NARRATIVE TYPE")
print("-" * 70)

results_df = val_df.copy()
results_df['predicted_score'] = pred_scores
results_df['true_score'] = true_scores
results_df['error'] = np.abs(pred_scores - true_scores)
results_df['residual'] = pred_scores - true_scores

for ntype in ['coherent', 'shuffled', 'repetitive', 'truncated']:
    subset = results_df[results_df['type'] == ntype]
    r2 = r2_score(subset['true_score'], subset['predicted_score'])
    
    print(f"\n{ntype.upper()}:")
    print(f"  Examples: {len(subset)}")
    print(f"  True mean: {subset['true_score'].mean():.3f} ± {subset['true_score'].std():.3f}")
    print(f"  Pred mean: {subset['predicted_score'].mean():.3f} ± {subset['predicted_score'].std():.3f}")
    print(f"  MAE: {subset['error'].mean():.3f}")
    print(f"  R²: {r2:.3f}")
    print(f"  Mean Residual: {subset['residual'].mean():.3f}")

# 4. Calibration Analysis
print("\n" + "="*70)
print("🔧 CALIBRATION ANALYSIS")
print("="*70)

from sklearn.linear_model import LinearRegression

# Fit calibration
calibrator = LinearRegression()
calibrator.fit(pred_scores.reshape(-1, 1), true_scores)

print(f"\nCalibration function: y = {calibrator.coef_[0]:.3f} * pred + {calibrator.intercept_:.3f}")

# Apply calibration
calibrated_scores = calibrator.predict(pred_scores.reshape(-1, 1))
calibrated_scores = np.clip(calibrated_scores, 0, 1)  # Ensure in [0, 1]

# Calculate calibrated metrics
cal_mse = mean_squared_error(true_scores, calibrated_scores)
cal_mae = mean_absolute_error(true_scores, calibrated_scores)
cal_r2 = r2_score(true_scores, calibrated_scores)
cal_corr = np.corrcoef(true_scores, calibrated_scores)[0, 1]
cal_within_0_2 = np.mean(np.abs(calibrated_scores - true_scores) < 0.2)

print("\n📈 METRICS COMPARISON:")
print("-" * 70)
print(f"                    Before      After       Target")
print(f"MSE:              {eval_results['eval_mse']:.4f}      {cal_mse:.4f}      0.015-0.025")
print(f"MAE:              {eval_results['eval_mae']:.4f}      {cal_mae:.4f}      0.08-0.12")
print(f"R²:               {eval_results['eval_r2_score']:.4f}     {cal_r2:.4f}      0.75-0.85")
print(f"Correlation:      {eval_results['eval_correlation']:.4f}      {cal_corr:.4f}      0.85-0.92")
print(f"Accuracy (±0.2):  {eval_results['eval_accuracy_0.2']:.4f}      {cal_within_0_2:.4f}      0.85-0.92")

# Visualize calibration
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Before calibration
axes[0].scatter(true_scores, pred_scores, alpha=0.3, s=20)
axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2)
axes[0].set_xlabel('True Score', fontsize=12)
axes[0].set_ylabel('Predicted Score', fontsize=12)
axes[0].set_title(f'Before Calibration (R²={eval_results["eval_r2_score"]:.3f})', 
                 fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# After calibration
axes[1].scatter(true_scores, calibrated_scores, alpha=0.3, s=20)
axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2)
axes[1].set_xlabel('True Score', fontsize=12)
axes[1].set_ylabel('Calibrated Score', fontsize=12)
axes[1].set_title(f'After Calibration (R²={cal_r2:.3f})', 
                 fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calibration_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Test on Examples
print("\n" + "="*70)
print("🧪 CALIBRATED PREDICTIONS ON TEST EXAMPLES")
print("="*70)

def predict_with_calibration(text):
    """Predict with calibration applied."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        raw_score = torch.sigmoid(outputs.logits).item()
        calibrated = calibrator.predict([[raw_score]])[0]
        calibrated = np.clip(calibrated, 0, 1)
    
    return raw_score, calibrated

test_examples = [
    ("The ancient library stretched endlessly before you, its towering shelves groaning under countless leather-bound tomes.", "High Quality"),
    ("You see a room. There is a door. There is a table.", "Low Quality"),
    ("The dragon roars. The dragon breathes fire. The dragon roars again. The dragon breathes more fire.", "Repetitive"),
    ("Your blade finds its mark with a satisfying thud. The orc's eyes widen in surprise before it crumples to", "Truncated"),
]

for text, category in test_examples:
    raw, calibrated = predict_with_calibration(text)
    print(f"\n{category}:")
    print(f"  Text: {text[:70]}...")
    print(f"  Raw Score: {raw:.3f}")
    print(f"  Calibrated Score: {calibrated:.3f}")

print("\n" + "="*70)
print("✓ DIAGNOSTIC COMPLETE")
print("="*70)
print("\nRECOMMENDATIONS:")
if cal_r2 > 0.7:
    print("✅ Calibration fixes the issue! Save the calibrator with your model.")
    print("   Use calibrated scores in production.")
else:
    print("⚠️  Calibration helps but not enough. Consider:")
    print("   1. Regenerating dataset with narrower score ranges")
    print("   2. Retraining with different loss function (Huber)")
    print("   3. Adjusting model architecture")

# 6. Save the calibrator
if cal_r2 > 0.7:
    print("\n" + "="*70)
    print("💾 SAVING CALIBRATOR")
    print("="*70)
    
    import pickle
    import json
    
    output_dir = CONFIG['output_dir']
    
    # Save calibrator
    calibrator_path = f"{output_dir}/calibrator.pkl"
    with open(calibrator_path, 'wb') as f:
        pickle.dump(calibrator, f)
    print(f"\n✓ Calibrator saved: {calibrator_path}")
    
    # Save calibration metadata
    cal_info = {
        'calibration_slope': float(calibrator.coef_[0]),
        'calibration_intercept': float(calibrator.intercept_),
        'calibrated_mae': float(cal_mae),
        'calibrated_r2': float(cal_r2),
        'calibrated_correlation': float(cal_corr),
        'original_mae': float(eval_results['eval_mae']),
        'original_r2': float(eval_results['eval_r2_score']),
        'improvement_mae': float(eval_results['eval_mae'] - cal_mae),
        'improvement_r2': float(cal_r2 - eval_results['eval_r2_score'])
    }
    
    with open(f"{output_dir}/calibration_info.json", 'w') as f:
        json.dump(cal_info, f, indent=2)
    print(f"✓ Calibration info saved: {output_dir}/calibration_info.json")
    
    # Save as numpy for other uses
    np.save(f"{output_dir}/calibration_params.npy", 
            np.array([calibrator.coef_[0], calibrator.intercept_]))
    print(f"✓ Calibration params saved: {output_dir}/calibration_params.npy")
    
    print("\n📦 FILES READY FOR PRODUCTION:")
    print(f"   {output_dir}/pytorch_model.bin")
    print(f"   {output_dir}/config.json")
    print(f"   {output_dir}/tokenizer.json")
    print(f"   {output_dir}/calibrator.pkl ⭐ NEW")
    print(f"   {output_dir}/calibration_info.json ⭐ NEW")
    
    print("\n✅ Model + Calibrator ready for deployment!")
    print("   See CALIBRATION_GUIDE.md for usage examples")
