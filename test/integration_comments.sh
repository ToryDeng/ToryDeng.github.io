#!/usr/bin/env bash
set -euo pipefail

tmp_dir="$(mktemp -d)"
tmp_override="${tmp_dir}/comments-test-override.yml"
tmp_site="${tmp_dir}/site"
giscus_fixture="_posts/2000-01-01-comments-integration-giscus.md"
disqus_fixture="_posts/2000-01-02-comments-integration-disqus.md"

if [[ -e "${giscus_fixture}" || -e "${disqus_fixture}" ]]; then
  echo "comments integration fixture already exists" >&2
  exit 1
fi

cleanup() {
  rm -f "${giscus_fixture}" "${disqus_fixture}"
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

cat >"${giscus_fixture}" <<'MARKDOWN'
---
layout: post
title: comments integration giscus
date: 2000-01-01 00:00:00
giscus_comments: true
related_posts: false
---

Giscus integration test fixture.
MARKDOWN

cat >"${disqus_fixture}" <<'MARKDOWN'
---
layout: post
title: comments integration disqus
date: 2000-01-02 00:00:00
disqus_comments: true
related_posts: false
---

Disqus integration test fixture.
MARKDOWN

cat >"${tmp_override}" <<'YAML'
giscus:
  repo: alshedivat/al-folio
  repo_id: R_kgDOExample
  category: Comments
  category_id: DIC_kwDOExample
YAML

bundle exec jekyll build --config "_config.yml,${tmp_override}" -d "${tmp_site}" >/dev/null

giscus_page="${tmp_site}/blog/2000/comments-integration-giscus/index.html"
disqus_page="${tmp_site}/blog/2000/comments-integration-disqus/index.html"

grep -q 'https://giscus.app/client.js' "${giscus_page}"
if grep -q 'giscus comments misconfigured' "${giscus_page}"; then
  echo "unexpected giscus misconfiguration warning in ${giscus_page}" >&2
  exit 1
fi

grep -q 'id="disqus_thread"' "${disqus_page}"
grep -q '.disqus.com/embed.js' "${disqus_page}"

echo "comments integration checks passed"
