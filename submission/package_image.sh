#!/usr/bin/env bash
# Export a built image as a gzipped classic docker-archive, the format the platform accepts. Recent Docker
# versions save OCI archives; those are converted with skopeo (from the saved file, not the daemon).
set -euo pipefail
IMAGE="${1:?usage: package_image.sh <image[:tag]> [out.tar.gz]}"
OUT="${2:-report2mri-submission.tar.gz}"
[[ "${IMAGE##*/}" == *:* ]] || IMAGE="$IMAGE:latest"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
docker save "$IMAGE" -o "$TMP/saved.tar"
if tar -tf "$TMP/saved.tar" | grep -qx 'index.json'; then
    skopeo copy --quiet "oci-archive:$TMP/saved.tar:${IMAGE##*:}" "docker-archive:$TMP/image.tar:$IMAGE"
    rm "$TMP/saved.tar"
else
    mv "$TMP/saved.tar" "$TMP/image.tar"
fi
if tar -tf "$TMP/image.tar" | grep -qE '^(index\.json|oci-layout)$'; then
    echo "conversion failed: the archive is still in OCI format" >&2
    exit 1
fi
gzip -c "$TMP/image.tar" > "$OUT"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
