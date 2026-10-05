#!/usr/bin/env bash
set -e

echo "args: $@"

MODULE_NAME=sciencebeam_trainer_delft.sequence_labelling.grobid_trainer

PROJECT=""
REGION=""
DISPLAY_NAME=""
CONTAINER_IMAGE_URI=""
MACHINE_TYPE="n1-highmem-8"
ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
ACCELERATOR_COUNT="1"
NO_ACCELERATOR=false
ENV_VARS=()

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --project)
            PROJECT="$2"; shift; shift ;;
        --region)
            REGION="$2"; shift; shift ;;
        --display-name)
            DISPLAY_NAME="$2"; shift; shift ;;
        --container-image-uri)
            CONTAINER_IMAGE_URI="$2"; shift; shift ;;
        --machine-type)
            MACHINE_TYPE="$2"; shift; shift ;;
        --accelerator-type)
            ACCELERATOR_TYPE="$2"; shift; shift ;;
        --accelerator-count)
            ACCELERATOR_COUNT="$2"; shift; shift ;;
        --no-accelerator)
            NO_ACCELERATOR=true; shift ;;
        --module-name)
            MODULE_NAME="$2"; shift; shift ;;
        --env)
            ENV_VARS+=("$2"); shift; shift ;;
        --)
            shift; break ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "${PROJECT}" ]; then echo "Error: --project is required"; exit 1; fi
if [ -z "${REGION}" ]; then echo "Error: --region is required"; exit 1; fi
if [ -z "${CONTAINER_IMAGE_URI}" ]; then echo "Error: --container-image-uri is required"; exit 1; fi

if [ -z "${DISPLAY_NAME}" ]; then
    DISPLAY_NAME="sciencebeam_$(date +%Y_%m_%d_%H%M%S -u)"
fi

if [ "${NO_ACCELERATOR}" = "true" ]; then
    WORKER_POOL_SPEC="machine-type=${MACHINE_TYPE},replica-count=1,container-image-uri=${CONTAINER_IMAGE_URI}"
else
    WORKER_POOL_SPEC="machine-type=${MACHINE_TYPE},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT},replica-count=1,container-image-uri=${CONTAINER_IMAGE_URI}"
fi

# the container entrypoint ends with `exec "$@"`, so these args are the command
# and `env` can carry variables into it; the args are joined with commas, so a
# value containing one cannot be passed this way
COMMAND=(python -m "${MODULE_NAME}")
if [ ${#ENV_VARS[@]} -gt 0 ]; then
    COMMAND=(env "${ENV_VARS[@]}" "${COMMAND[@]}")
fi

ARGS_CSV=$(printf '%s\n' "${COMMAND[@]}" "$@" | paste -sd ',')

echo "DISPLAY_NAME: ${DISPLAY_NAME}"
echo "CONTAINER_IMAGE_URI: ${CONTAINER_IMAGE_URI}"
echo "WORKER_POOL_SPEC: ${WORKER_POOL_SPEC}"
echo "ARGS_CSV: ${ARGS_CSV}"
echo ""

gcloud ai custom-jobs create \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --display-name="${DISPLAY_NAME}" \
    --worker-pool-spec="${WORKER_POOL_SPEC}" \
    --args="${ARGS_CSV}"
