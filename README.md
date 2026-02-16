# BioNet Compare — Streamlit app to compare multiple biological networks

## Quick start

### macOS/Linux
```bash
bash install.sh
source .venv/bin/activate
streamlit run app.py
```

### Windows (PowerShell)
```powershell
.\install_windows.ps1
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

If you see **StreamlitDuplicateElementId**, ensure every `st.download_button()` has a unique `key=` (already fixed in this version).
