# Heart Diet Flask Project Rebuild - TODO List

## Approved Plan Breakdown (Step-by-step execution)

### 1. Create static subdirectories and reorganize files
- [x] Create static/css/, static/js/, static/images/
- [x] Move static/styles.css → static/css/styles.css
- [x] Move static/pics/* → static/images/
- [ ] Update ALL HTML paths: 'filename=pics/' → 'filename=images/', 'filename=styles.css' → 'filename=css/styles.css'

### 2. Fix HTML files
- [ ] Standardize checkup.html: extend base.html, Bootstrap5, remove old CDN/inline styles, fix paths
- [ ] Verify other HTML paths correct (search confirms most good)

### 3. Rebuild app.py
- [x] Add full routes: /home, /diet, /bmi, /checkup, /about, /fitness, /sleep, /stress, /QuitSmoking etc.
- [x] Global model load from models/ or graceful fail
- [x] Enhance /result: proper prediction + render result template
- [x] Add stub diet logic for /diet_results

### 4. Clean requirements.txt
- [x] Set exact deps: flask, gunicorn, numpy, pandas, scikit-learn, matplotlib

### 5. Handle ML files
- [ ] Create placeholder models/heart_model.pkl if needed (mock)
- [ ] Create pickle_files/ if required by old code

### 6. Remove junk (none found)
- [ ] Delete test_csv.py if exists

### 7. Test locally
- [ ] python app.py → check /, /test, /result POST
- [ ] gunicorn app:app → Render-ready

### 8. Git commit & push
- [ ] git add . && git commit && git push

### 9. Verify Render deployment ready

**Progress: Ready to start Step 1**

