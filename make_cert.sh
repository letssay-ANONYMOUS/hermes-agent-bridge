#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p certs

ip_addr="${1:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)}"

cat > certs/openssl.cnf <<CONFOF
[req]
distinguished_name=req_distinguished_name
x509_extensions=v3_req
prompt=no

[req_distinguished_name]
CN=Hermes Voice Room

[v3_req]
keyUsage=digitalSignature, keyEncipherment, dataEncipherment
extendedKeyUsage=serverAuth
subjectAltName=@alt_names

[alt_names]
DNS.1=localhost
IP.1=127.0.0.1
IP.2=${ip_addr}
CONFOF

openssl req -x509 -nodes -days 825 -newkey rsa:2048 \
  -keyout certs/voice-room.key \
  -out certs/voice-room.crt \
  -config certs/openssl.cnf

echo "Created cert for https://${ip_addr}:8765"
