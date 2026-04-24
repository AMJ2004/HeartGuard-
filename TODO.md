# Fix Flask BuildError & Stabilize App

- [x] Analyze root cause and broken endpoints
- [x] Edit `app.py` — add missing routes (`home`, `about`, `stress`, `fitness`, `sleep`, `quit_smoking`, `diet_recommendations`) and remove `/home` from `index()`
- [x] Edit `templates/base.html` — fix navbar AI Assessment link, fix bare `url_for('quit_smoking')` in footer
- [x] Edit `templates/heart.html` — fix bare `url_for('quit_smoking')` in Smart Quit Programs card
- [x] Edit `templates/heartdisease.html` — fix raw `url_for('quit_smoking')` text
- [x] Fix `templates/assessment.html` Jinja2 syntax error (`endblock scripts`)
- [x] Run local Flask test
- [x] Git commit & push

