import pytest
from simple_logger.logger import get_logger
from typing import Any
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from utilities.constants import KServeDeploymentType, Protocols
from tests.model_serving.model_runtime.mlserver.constant import INPUT_QUERY, MODEL_PATH_PREFIX
from tests.model_serving.model_runtime.mlserver.constant import BASE_RAW_DEPLOYMENT_CONFIG, BASE_SEVERRLESS_DEPLOYMENT_CONFIG
from tests.model_serving.model_runtime.mlserver.utils import validate_inference_request


LOGGER = get_logger(name=__name__)

MODEL_NAME: str = "sklearn-iris"
MODEL_VERSION: str = "v1.0.0"
MODEL_PATH: str = f"{MODEL_PATH_PREFIX}/sklearn"


pytestmark = pytest.mark.usefixtures("valid_aws_config", "mlserver_rest_serving_runtime_template", "mlserver_grpc_serving_runtime_template")


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, mlserver_inference_service",
    [
        pytest.param(
            {   "name": "sklearn-iris-raw-rest" },
            {   "model-dir": MODEL_PATH         },
            {
                "deployment_type": KServeDeploymentType.RAW_DEPLOYMENT, 
                "protocol": Protocols.REST,
            },
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "name": MODEL_NAME,
            },
            id="sklearn-iris-raw-rest-deployment",
        ),
        pytest.param(
            {   "name": "sklearn-iris-serverless-rest"  },
            {   "model-dir": MODEL_PATH                 },
            {
                "deployment_type": KServeDeploymentType.SERVERLESS,
                "protocol": Protocols.REST,
            },
            {
                **BASE_SEVERRLESS_DEPLOYMENT_CONFIG,
                "name": MODEL_NAME,
            },
            id="sklearn-iris-serverless-rest-deployment",
        ),
    ],
    indirect=True,
)
class TestSkLearnModel:
    def test_sklearn_model_inference(
        self,
        mlserver_inference_service: InferenceService,
        mlserver_pod_resource: Pod,
        response_snapshot: Any,
    ):
        validate_inference_request(
            pod_name=mlserver_pod_resource.name,
            isvc=mlserver_inference_service,
            response_snapshot=response_snapshot,
            input_query=INPUT_QUERY,
            model_version=MODEL_VERSION
        )