# Fix Flask Project - TODO

## Step 1: Fix Missing Routes
- [x] /home → render checkup.html (was redirecting to index)
- [x] /about → render about.html (already exists)
- [x] /fitness → render fitness.html (already exists)
- [x] /sleep → render sleep.html (already exists)
- [x] /stress → render stress.html (already exists)
- [x] /QuitSmoking → render QuitSmoking.html (add alongside /quit_smoking)

## Step 2: Fix Result Flow
- [x] /result → conditional render heartdisease_detected.html (prediction==1) or nodisease.html (prediction==0)
- [x] Pass required variables (age, gender, bmi, sysBP, diaBP, glucose, totChol/CHol, p, b, d)
- [x] Include prediction in session user_data for diet recommender

## Step 3: Fix 'int not iterable' Error
- [x] Search for `{% for x in prediction %}` → None found (already clean)

## Step 4: Fix Diet Flow
- [x] /diet → render diet.html (info/landing page)
- [x] Move diet logic to /diet_results route
- [x] Update diet.html CTA links to point to /diet_results

## Step 5: Fix Static Paths
- [x] Search for bare `static/...` → None found (already uses url_for)

## Step 6: Clean Project
- [x] Delete templates/home.html (duplicate)
- [x] Delete templates/Normal.html (unused)
- [x] Delete templates/Underweight.html (unused)
- [x] Delete templates/Overweight.html (unused)

## Step 7: Test Flow
- [x] Homepage loads
- [x] AI Assessment opens
- [x] Form submits
- [x] Result shows
- [x] Diet page works

## Step 8: Commit
- [x] git add .
- [x] git commit -m "Fixed routes, template errors, and prediction flow"
- [x] git push origin main

