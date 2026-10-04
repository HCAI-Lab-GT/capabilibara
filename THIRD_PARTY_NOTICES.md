# Third-party dependencies

The `scripts/bootstrap_local_deps.sh` script checks out pinned revisions of
[Bergson](https://github.com/EleutherAI/bergson) and
[OLMES](https://github.com/allenai/olmes), then applies the patches in
`patches/`. Neither upstream source tree is included in this repository.

The Bergson patch modifies MIT-licensed upstream code. Its license and
copyright notice are retained in [`third_party/BERGSON-LICENSE`](third_party/BERGSON-LICENSE).
The OLMES patch modifies Apache-2.0-licensed upstream code. Its license is
retained in [`third_party/OLMES-LICENSE`](third_party/OLMES-LICENSE).
