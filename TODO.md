# Fix Flask ML Pipeline - TODO

## Plan
- [x] 1. Fix `app.py` — Load model, scaler, threshold at startup (not per-request)
- [x] 2. Fix `app.py` — Use EXACT feature order: ['sysBP','glucose','age','totChol','diaBP','prevalentHyp','diabetes','male','BPMeds','BMI']
- [x] 3. Fix `app.py` — Compute derived features correctly from inputs
- [x] 4. Fix `app.py` — Add BMI validation
- [x] 5. Fix `app.py` — Scale data with scaler, use probability + threshold
- [x] 6. Fix `templates/heartdisease_detected.html` — Add High Risk / Low Risk indicator
- [ ] 7. Test the flow
- [ ] 8. Commit and push

