# Astral's image is the official python image plus uv, so Python and uv are
# pinned in one line. Use the trixie tags, the bookworm tags are no longer built.
FROM ghcr.io/astral-sh/uv:0.13.5-python3.12-trixie AS build

WORKDIR /usr/src/app

ARG UV_INDEX
ENV UV_INDEX=$UV_INDEX

ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV="/usr/src/app/.venv"

COPY . .
RUN make install

# Same Debian release as the build stage.
FROM python:3.12-slim-trixie AS main

# procps is required by Nextflow
RUN apt-get update -y \
    && apt-get install -y -o Acquire::Retries=10 --no-install-recommends \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Nothing here uses pip, and the one the image ships has an old CA bundle
# that 3.12 will not update (docker-library/python issue 1142).
RUN python -m pip uninstall -y pip

WORKDIR /usr/src/app

ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV="/usr/src/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ARG BUILD_DATE
ARG VCS_REF
ARG BUILD_VERSION

LABEL com.oncoimmunity.build-date=$BUILD_DATE
LABEL com.oncoimmunity.vcs-url="https://github.com/OncoImmunity/her2_stopping_rule"
LABEL com.oncoimmunity.vcs-ref=$VCS_REF
LABEL com.oncoimmunity.version=$BUILD_VERSION

COPY --from=build $VIRTUAL_ENV $VIRTUAL_ENV

CMD ["her2_stopping_rule", "--help"]
