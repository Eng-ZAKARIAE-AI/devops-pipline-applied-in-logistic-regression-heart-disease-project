#!/usr/bin/env python3
"""Build script for UI - injects environment config into index.html"""
import os
import re
from pathlib import Path

def load_env(env_path):
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip()
    return env

def main():
    ui_dir = Path(__file__).parent
    env = load_env(ui_dir / '.env')
    
    # Determine API base URL
    env_mode = env.get('ENV_MODE', 'local')
    api_base = env.get('API_BASE_URL', 'http://localhost:5000')
    
    # Allow override via environment variable (for CI/CD)
    api_base = os.environ.get('API_BASE_URL', api_base)
    
    # Read template
    template_path = ui_dir / 'index.html'
    html = template_path.read_text()
    
    # Replace placeholder
    config_js = f"""<script id="app-config">
  window.APP_CONFIG = {{
    API_BASE: '{api_base}',
    ENV_MODE: '{env_mode}'
  }};
</script>"""
    
    # Insert before the first script tag
    html = html.replace(
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>',
        config_js + '\n<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
    )
    
    # Also inject a small script to auto-fill the API input if empty
    inject_script = f"""
<script>
  (function() {{
    const cfg = window.APP_CONFIG;
    if (cfg && cfg.API_BASE) {{
      const input = document.getElementById('api');
      if (input && !input.value) {{
        input.value = cfg.API_BASE.replace(/\\/predict\\/?$/, '') + '/predict';
      }}
    }}
  }})();
</script>
"""
    # Insert before the closing body tag
    html = html.replace('</body>', inject_script + '\n</body>')
    
    # Write built version
    dist_dir = ui_dir / 'dist'
    dist_dir.mkdir(exist_ok=True)
    (dist_dir / 'index.html').write_text(html)
    
    # Copy any other static assets
    for f in ui_dir.glob('*.png'):
        import shutil
        shutil.copy2(f, dist_dir / f.name)
    
    print(f"Built UI to {dist_dir}")
    print(f"  ENV_MODE={env_mode}")
    print(f"  API_BASE={api_base}")

if __name__ == '__main__':
    main()