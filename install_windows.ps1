$ErrorActionPreference = "Stop"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "✅ Installation complete"
Write-Host "Run: .\\.venv\\Scripts\\Activate.ps1 ; streamlit run app.py"
