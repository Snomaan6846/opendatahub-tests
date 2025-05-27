from typing import Any, Generator
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.template import Template
from ocp_resources.service_account import ServiceAccount
from tests.model_serving.model_runtime.mlserver.utils import (
    kserve_s3_endpoint_secret,
    skip_if_deployment_mode,
)
import pytest
from pytest import FixtureRequest
from syrupy.extensions.json import JSONSnapshotExtension
from tests.model_serving.model_runtime.mlserver.constant import PREDICT_RESOURCES, TEMPLATE_MAP, TEMPLATE_FILE_PATH
from utilities.constants import KServeDeploymentType, Labels, RuntimeTemplates, Protocols
from simple_logger.logger import get_logger

from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label
from utilities.serving_runtime import ServingRuntimeFromTemplate
from pytest_testconfig import config as py_config


LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def mlserver_grpc_serving_runtime_template(admin_client: DynamicClient) -> Template:
    grpc_template_yaml = TEMPLATE_FILE_PATH.get(Protocols.GRPC)
    with Template(
        client=admin_client,
        yaml_file=grpc_template_yaml,
        namespace=py_config["applications_namespace"],
    ) as tp:
        yield tp


@pytest.fixture(scope="class")
def mlserver_rest_serving_runtime_template(admin_client: DynamicClient) -> Template:
    rest_template_yaml = TEMPLATE_FILE_PATH.get(Protocols.REST)
    with Template(
        client=admin_client,
        yaml_file=rest_template_yaml,
        namespace=py_config["applications_namespace"],
    ) as tp:
        yield tp


@pytest.fixture(scope="class")
def serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_runtime_image: str,
) -> Generator[ServingRuntime, None, None]:
    protocol = request.param["protocol"].lower()
    template_name = TEMPLATE_MAP.get(protocol, RuntimeTemplates.MLSERVER_REST)
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="mlserver-runtime",
        namespace=model_namespace.name,
        template_name=template_name,
        deployment_type=request.param["deployment_type"],
        runtime_image=mlserver_runtime_image,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def mlserver_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    isvc_kwargs = {
        "client": admin_client,
        "name": request.param["name"],
        "namespace": model_namespace.name,
        "runtime": serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": model_service_account.name,
        "deployment_mode": request.param.get("deployment_mode", KServeDeploymentType.RAW_DEPLOYMENT),
    }
    gpu_count = request.param.get("gpu_count", 0)
    timeout = request.param.get("timeout")
    identifier = Labels.Nvidia.NVIDIA_COM_GPU
    resources: Any = PREDICT_RESOURCES["resources"]
    resources["requests"][identifier] = gpu_count
    resources["limits"][identifier] = gpu_count
    isvc_kwargs["resources"] = resources
    if timeout:
        isvc_kwargs["timeout"] = timeout
    if gpu_count > 1:
        isvc_kwargs["volumes"] = PREDICT_RESOURCES["volumes"]
        isvc_kwargs["volumes_mounts"] = PREDICT_RESOURCES["volume_mounts"]

    if min_replicas := request.param.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def model_service_account(admin_client: DynamicClient, kserve_endpoint_s3_secret: Secret) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=kserve_endpoint_s3_secret.namespace,
        name="models-bucket-sa",
        secrets=[{"name": kserve_endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def kserve_endpoint_s3_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Secret:
    with kserve_s3_endpoint_secret(
        admin_client=admin_client,
        name="models-bucket-secret",
        namespace=model_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture
def response_snapshot(snapshot: Any) -> Any:
    return snapshot.use_extension(extension_class=JSONSnapshotExtension)


@pytest.fixture
def mlserver_pod_resource(admin_client: DynamicClient, mlserver_inference_service: InferenceService) -> Pod:
    return get_pods_by_isvc_label(client=admin_client, isvc=mlserver_inference_service)[0]


@pytest.fixture
def skip_if_serverless_deployemnt(mlserver_inference_service: InferenceService) -> None:
    skip_if_deployment_mode(
        isvc=mlserver_inference_service,
        deployment_type=KServeDeploymentType.SERVERLESS,
        deployment_message="Test is being skipped because model is being deployed in serverless mode",
    )


@pytest.fixture
def skip_if_raw_deployemnt(mlserver_inference_service: InferenceService) -> None:
    skip_if_deployment_mode(
        isvc=mlserver_inference_service,
        deployment_type=KServeDeploymentType.RAW_DEPLOYMENT,
        deployment_message="Test is being skipped because model is being deployed in raw mode",
    )
