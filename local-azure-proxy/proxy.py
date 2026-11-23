"""Proxy local OpenAI-compatible -> Azure OpenAI v1, pour tester Auto Ontology.

- Retire le préfixe ``azure/`` / ``openai/`` du champ ``model`` (Auto Ontology l'exige
  pour choisir ChatOpenAI, Azure attend le nom de déploiement nu).
- Embeddings : retire les champs propres à NVIDIA (input_type, truncate, modality)
  et complète chaque vecteur par des zéros jusqu'à PAD_DIM (Auto Ontology crée ses
  colonnes en vector(2048), ada-002 en produit 1536 ; le cosinus est inchangé).
- La clé Azure reste ici (AZURE_OPENAI_KEY) : Auto Ontology n'en voit qu'une factice.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1"
KEY = os.environ["AZURE_OPENAI_KEY"]
PAD_DIM = int(os.environ.get("PAD_DIM", "2048"))
PREFIXES = ("azure/", "openai/", "aws/")
NVIDIA_ONLY = ("input_type", "truncate", "modality")


def _strip(model: str) -> str:
    for p in PREFIXES:
        if model.startswith(p):
            return model[len(p) :]
    return model


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("proxy %s\n" % (fmt % args))

    def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/health", "/v1/health"):
            self._send(200, b'{"ok":true}')
        else:
            self._send(404, b'{"error":"not proxied"}')

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path.startswith("/v1/"):
            path = path[3:]
        if path not in ("/chat/completions", "/embeddings"):
            self._send(404, json.dumps({"error": f"not proxied: {path}"}).encode())
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.loads(raw or b"{}")
        if "model" in body:
            body["model"] = _strip(body["model"])
        is_embed = path == "/embeddings"
        if is_embed:
            for k in NVIDIA_ONLY:
                body.pop(k, None)
        req = urllib.request.Request(
            UPSTREAM + path,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "api-key": KEY},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read())
            return
        except Exception as e:  # réseau
            self._send(502, json.dumps({"error": str(e)}).encode())
            return

        ctype = resp.headers.get("Content-Type", "application/json")
        if "text/event-stream" in ctype:  # streaming de chat : relais tel quel
            self.send_response(resp.status)
            self.send_header("Content-Type", ctype)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while chunk := resp.read1(65536):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            return

        data = resp.read()
        if is_embed and resp.status == 200:
            payload = json.loads(data)
            for item in payload.get("data", []):
                vec = item.get("embedding")
                if isinstance(vec, list) and len(vec) < PAD_DIM:
                    item["embedding"] = vec + [0.0] * (PAD_DIM - len(vec))
            data = json.dumps(payload).encode()
        self._send(resp.status, data, ctype)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    sys.stderr.write(f"proxy -> {UPSTREAM} on :{port}, pad={PAD_DIM}\n")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
