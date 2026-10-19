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
SCHEDULING_STRATEGY=""
MAX_WAIT_DURATION=""

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
        --scheduling-strategy)
            SCHEDULING_STRATEGY="$2"; shift; shift ;;
        --max-wait-duration)
            MAX_WAIT_DURATION="$2"; shift; shift ;;
        --)
            shift; break ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "${PROJECT}" ]; then echo "Error: --project is required"; exit 1; fi
if [ -z "${REGION}" ]; then echo "Error: --region is required"; exit 1; fi
if [ -z "${CONTAINER_IMAGE_URI}" ]; then echo "Error: --container-image-uri is required"; exit 1; fi

case "${SCHEDULING_STRATEGY}" in
    ""|STANDARD|SPOT|FLEX_START) ;;
    *)
        echo "Error: --scheduling-strategy must be STANDARD, SPOT or FLEX_START, not '${SCHEDULING_STRATEGY}'"
        exit 1 ;;
esac

if [ -n "${MAX_WAIT_DURATION}" ] && [ "${SCHEDULING_STRATEGY}" != "FLEX_START" ]; then
    echo "Error: --max-wait-duration only applies to --scheduling-strategy FLEX_START"
    exit 1
fi

if [ -z "${DISPLAY_NAME}" ]; then
    DISPLAY_NAME="sciencebeam_$(date +%Y_%m_%d_%H%M%S -u)"
fi

# The job spec goes in a file rather than through --worker-pool-spec, because
# that flag carries the machine, the accelerator and the image and nothing else:
# no environment variables, and no scheduling strategy. Both are fields of the
# API, reachable only this way. A list of args also stops a value containing a
# comma from being split, which the flag's csv could not express.
JOB_SPEC_FILE="$(mktemp -t sciencebeam-job-spec-XXXXXX.yaml)"
trap 'rm -f "${JOB_SPEC_FILE}"' EXIT

# json is valid yaml, and emitting it keeps every value quoted correctly
ENV_VARS_JOINED=""
if [ ${#ENV_VARS[@]} -gt 0 ]; then
    ENV_VARS_JOINED="$(printf '%s\n' "${ENV_VARS[@]}")"
fi
export MACHINE_TYPE ACCELERATOR_TYPE ACCELERATOR_COUNT NO_ACCELERATOR \
    CONTAINER_IMAGE_URI MODULE_NAME ENV_VARS_JOINED SCHEDULING_STRATEGY MAX_WAIT_DURATION

python3 - "${JOB_SPEC_FILE}" "$@" <<'PYTHON' || { echo "Error: could not write the job spec"; exit 1; }
import json, os, sys

spec_file, trainer_args = sys.argv[1], sys.argv[2:]
machine_spec = {"machineType": os.environ["MACHINE_TYPE"]}
if os.environ["NO_ACCELERATOR"] != "true":
    machine_spec["acceleratorType"] = os.environ["ACCELERATOR_TYPE"]
    machine_spec["acceleratorCount"] = int(os.environ["ACCELERATOR_COUNT"])

container_spec = {
    "imageUri": os.environ["CONTAINER_IMAGE_URI"],
    "args": ["python", "-m", os.environ["MODULE_NAME"], *trainer_args],
}

env_vars = [value for value in os.environ.get("ENV_VARS_JOINED", "").split("\n") if value]
if env_vars:
    container_spec["env"] = [
        {"name": name, "value": value}
        for name, _, value in (item.partition("=") for item in env_vars)
    ]

spec = {
    "workerPoolSpecs": [
        {"machineSpec": machine_spec, "replicaCount": 1, "containerSpec": container_spec}
    ]
}

strategy = os.environ.get("SCHEDULING_STRATEGY", "")
if strategy:
    scheduling = {"strategy": strategy}
    if os.environ.get("MAX_WAIT_DURATION"):
        scheduling["maxWaitDuration"] = os.environ["MAX_WAIT_DURATION"]
    spec["scheduling"] = scheduling

with open(spec_file, "w", encoding="utf-8") as out:
    json.dump(spec, out, indent=2)
PYTHON

echo "DISPLAY_NAME: ${DISPLAY_NAME}"
echo "CONTAINER_IMAGE_URI: ${CONTAINER_IMAGE_URI}"
echo "MACHINE_TYPE: ${MACHINE_TYPE}"
echo "ACCELERATOR: $([ "${NO_ACCELERATOR}" = "true" ] && echo none || echo "${ACCELERATOR_TYPE} x${ACCELERATOR_COUNT}")"
echo "SCHEDULING: ${SCHEDULING_STRATEGY:-default}${MAX_WAIT_DURATION:+ (max wait ${MAX_WAIT_DURATION})}"
echo "JOB SPEC:"
cat "${JOB_SPEC_FILE}"
echo ""

gcloud ai custom-jobs create \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --display-name="${DISPLAY_NAME}" \
    --config="${JOB_SPEC_FILE}"
