FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
COPY target/release/ds4-flash-rs /usr/local/bin/ds4-flash-rs
COPY THIRD_PARTY_NOTICES.md /usr/share/doc/ds4-flash-rs/THIRD_PARTY_NOTICES.md
ENTRYPOINT ["/usr/local/bin/ds4-flash-rs"]
