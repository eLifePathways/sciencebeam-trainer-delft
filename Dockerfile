FROM ghcr.io/astral-sh/uv:python3.11-bookworm AS dev

# # install gcloud to make it easier to access cloud storage
# RUN mkdir -p /usr/local/gcloud \
#     && curl -q https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz -o /tmp/google-cloud-sdk.tar.gz \
#     && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
#     && rm /tmp/google-cloud-sdk.tar.gz \
#     && /usr/local/gcloud/google-cloud-sdk/install.sh --usage-reporting false \
#     && /usr/local/gcloud/google-cloud-sdk/bin/gcloud components install --quiet beta

# ENV PATH /usr/local/gcloud/google-cloud-sdk/bin:$PATH

# install wapiti
ARG wapiti_source_download_url
RUN if [ ! -z "${wapiti_source_download_url}" ]; then \
    curl --location -q "${wapiti_source_download_url}" -o /tmp/wapiti.tar.gz \
    && ls -lh /tmp/wapiti.tar.gz \
    && mkdir -p /tmp/wapiti \
    && tar --strip-components 1 -C /tmp/wapiti -xvf /tmp/wapiti.tar.gz \
    && rm /tmp/wapiti.tar.gz \
    && cd /tmp/wapiti \
    && make \
    && make install \
    && rm -rf /tmp/wapiti; \
    fi

ENV PROJECT_FOLDER=/opt/sciencebeam-trainer-delft

WORKDIR ${PROJECT_FOLDER}

ENV VENV=/opt/venv
ENV VIRTUAL_ENV=${VENV} PYTHONUSERBASE=${VENV} PATH=${VENV}/bin:$PATH

RUN uv venv "${VENV}"


# cpu or gpu: the two torch extras conflict, so exactly one is named. The
# image pushed for GPU training is built with --build-arg torch_extra=gpu
ARG torch_extra=cpu

COPY pyproject.toml uv.lock ./
RUN uv sync --active --frozen \
    --extra delft --extra gcs --extra "${torch_extra}" \
    --all-groups

COPY sciencebeam_trainer_delft ./sciencebeam_trainer_delft
COPY README.md ./

COPY delft ./delft

COPY .flake8 .pylintrc pytest.ini ./
COPY tests ./tests


# python-dist-builder
FROM dev AS python-dist-builder

ARG python_package_version
RUN echo "Setting version to: $version" && \
    uv version "$python_package_version"
RUN uv build && \
    ls -l dist


# python-dist
FROM scratch AS python-dist

WORKDIR /dist

COPY --from=python-dist-builder /opt/sciencebeam-trainer-delft/dist /dist


# lint-flake8
FROM dev AS lint-flake8

RUN python -m flake8 sciencebeam_trainer_delft tests


# lint-pylint
FROM dev AS lint-pylint

RUN python -m pylint sciencebeam_trainer_delft tests


# lint-mypy
FROM dev AS lint-mypy

RUN python -m mypy --ignore-missing-imports sciencebeam_trainer_delft tests


# pytest-not-slow
FROM dev AS pytest-not-slow

RUN python -m pytest -p no:cacheprovider -m 'not slow'


# pytest-slow
FROM dev AS pytest-slow

RUN python -m pytest -p no:cacheprovider -m 'slow'


# main image
FROM dev AS delft

# On Vertex AI (and similar GKE-based GPU infrastructure), the host's NVIDIA driver
# libraries (e.g. libcuda.so) are bind-mounted into the container at /usr/local/nvidia,
# rather than being auto-discovered the way newer CDI-based Docker GPU setups do. Our
# base image has no CUDA-awareness, so it never adds this path; without it, torch
# can't find libcuda.so even though the host driver is present, and falls back to CPU.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:$LD_LIBRARY_PATH

# add additional wrapper entrypoint for OVERRIDE_EMBEDDING_URL
COPY ./docker/entrypoint.sh ${PROJECT_FOLDER}/entrypoint.sh
ENTRYPOINT ["/opt/sciencebeam-trainer-delft/entrypoint.sh"]
CMD ["bash"]
