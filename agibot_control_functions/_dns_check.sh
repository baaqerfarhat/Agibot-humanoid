#!/usr/bin/env bash
for h in ollama.com registry.ollama.ai huggingface.co hf.co cdn-lfs.huggingface.co github.com objects.githubusercontent.com raw.githubusercontent.com pypi.org files.pythonhosted.org download.pytorch.org; do
  if getent hosts "$h" >/dev/null 2>&1; then
    ip=$(getent hosts "$h" | head -1 | awk '{print $1}')
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 "https://$h" 2>/dev/null)
    echo "OK    $h -> $ip  (HTTP $code)"
  else
    echo "FAIL  $h  (no DNS)"
  fi
done
