FROM ubuntu:24.04
COPY target/release/ds4-flash-rs /usr/local/bin/ds4-flash-rs
ENTRYPOINT ["/usr/local/bin/ds4-flash-rs"]
