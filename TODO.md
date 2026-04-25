# UI Routing Fix TODO

- [x] app.py: /assessment → checkup.html, remove /home, rename diet() → diet_preferences()
- [x] templates/checkup.html: overwrite with assessment.html content (AI form)
- [x] templates/base.html: fix url_for('home') → url_for('assessment'), url_for('diet') → url_for('diet_preferences')
- [x] templates/about.html: fix url_for('home') → url_for('assessment')
- [x] templates/fitness.html: fix url_for('home') → url_for('assessment')
- [x] templates/heartdisease.html: fix url_for('home') → url_for('assessment')
- [x] templates/nodisease.html: fix url_for('home') → url_for('assessment'), url_for('diet') → url_for('diet_preferences')
- [x] templates/heartdisease_detected.html: fix url_for('diet') → url_for('diet_preferences')
- [x] templates/sleep.html: fix url_for('home') → url_for('assessment')
- [x] templates/stress.html: fix url_for('home') → url_for('assessment')
- [x] templates/QuitSmoking.html: fix url_for('home') → url_for('assessment')
- [x] templates/medical_filters.html: fix url_for('diet') → url_for('diet_preferences')
- [x] Delete templates/assessment.html
- [x] Delete templates/analysis.html
- [x] Test & git commit/push

