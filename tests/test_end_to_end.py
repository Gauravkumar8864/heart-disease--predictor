from pathlib import Path

def test_repo_layout():
    required = [
        'README.md',
        'requirements.txt',
        'src/train.py',
        'src/preprocess.py',
        'app/gradio_app.py',
        'LICENSE',
        '.gitignore'
    ]
    for rel in required:
        assert Path(rel).exists(), f"Missing file: {rel}"