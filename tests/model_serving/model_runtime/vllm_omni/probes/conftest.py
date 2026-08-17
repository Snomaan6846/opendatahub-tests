"""Fixtures for vLLM-Omni probes tests."""

from collections.abc import Generator
from copy import deepcopy
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.vllm.utils import dedupe_vllm_cli_args
from tests.model_serving.model_runtime.vllm_omni.constant import (
    OMNI_ISVC_TIMEOUTS,
    OMNI_MULTI_GPU_RESOURCES,
    OMNI_SERVING_ARGS,
    OMNI_SINGLE_GPU_RESOURCES,
    OMNI_VOLUME_MOUNTS,
    OMNI_VOLUMES,
    QWEN3_OMNI_MODEL_PATH,
)
from utilities.constants import KServeDeploymentType, Labels
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label


@pytest.fixture(scope="function")
def vllm_omni_probes_pod_resource(
    admin_client: DynamicClient, vllm_omni_probes_inference_service: InferenceService
) -> Pod:
    """Function-scoped predictor pod for the probes omni_isvc InferenceService."""
    pods = list(get_pods_by_isvc_label(client=admin_client, isvc=vllm_omni_probes_inference_service))
    assert pods, f"No predictor pods found for ISVC {vllm_omni_probes_inference_service.name}"
    return pods[0]


@pytest.fixture(scope="class")
def vllm_omni_probes_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    vllm_omni_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    vllm_omni_model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    """Class-scoped vLLM-Omni InferenceService that waits for Ready.

    Resources auto-scale based on gpu_count: multi-GPU uses heavy spec,
    single-GPU uses lighter spec matching vLLM defaults.
    """
    gpu_count = request.param.get("gpu_count", 1)
    base_resources = OMNI_MULTI_GPU_RESOURCES if gpu_count > 1 else OMNI_SINGLE_GPU_RESOURCES
    resources = deepcopy(x=base_resources["resources"])
    resources["requests"][Labels.Nvidia.NVIDIA_COM_GPU] = gpu_count
    resources["limits"][Labels.Nvidia.NVIDIA_COM_GPU] = gpu_count

    serving_args = list(OMNI_SERVING_ARGS)

    with create_isvc(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        runtime=vllm_omni_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=vllm_omni_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=vllm_omni_model_service_account.name,
        deployment_mode=request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        external_route=True,
        resources=resources,
        volumes=OMNI_VOLUMES,
        volumes_mounts=OMNI_VOLUME_MOUNTS,
        argument=dedupe_vllm_cli_args(arguments=serving_args),
        timeout=request.param.get(
            "timeout",
            OMNI_ISVC_TIMEOUTS.get(request.param.get("model_path", QWEN3_OMNI_MODEL_PATH), 1800),
        ),
    ) as isvc:
        yield isvc
