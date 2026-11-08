# Xunsearch provenance

The unmodified PHP SDK files under `vendor/xunsearch/lib/` come from
<https://github.com/hightman/xunsearch>, commit
`6feed30a57732341412434ba6ab2224332dd62d8`. Upstream identifies this source
as Xunsearch 1.4.17. The directory retains the upstream `COPYING`, `AUTHORS`,
and SDK documentation license. Original copyright headers remain intact.

The Docker build downloads the official full source distribution from
<https://www.xunsearch.com/download/xunsearch-full-latest.tar.bz2> and checks
SHA256 `2ed66562f5cc6350e084c50edffa970c5306d348ae7c1da7ba9aa140afec7743`
before compilation. The verified archive contains `xunsearch-full-1.4.17`.
If upstream replaces the archive, the checksum check fails; update the URL
and checksum deliberately after validating the replacement.

This integration uses Xunsearch/Xapian BM25 and SCWS tokenization. It does
not implement the original hosted dense-retrieval model or neural reranker.
Powered by Xunsearch: <https://www.xunsearch.com>.
