# Heart Diet Flask Fixes - TODO

## Plan Steps (3 Issues Only):
1. ✅ Create TODO.md
2. ✅ Add `/diet-preferences` route to app.py
3. ✅ Fix `/bmi` route in app.py (added height > 0 check, exact task spec)
4. ✅ Test: Server running at http://127.0.0.1:5000 (model fallback OK)
5. ☐ Git commit/push

**Status**: COMPLETE! All 3 issues fixed:
- ✅ Diet_preferences: Route added, /diet-preferences loads diet_preferences.html
- ✅ BMI: Fixed server error (height>0 check, safe float parsing)
- ✅ Images: Already using {{ url_for('static', filename='...') }} everywhere

**Verification**:
- App runs successfully
- No more URL errors
- BMI handles invalid input/height=0
- Static files load correctly via base.html

Ready for git commit/push.
