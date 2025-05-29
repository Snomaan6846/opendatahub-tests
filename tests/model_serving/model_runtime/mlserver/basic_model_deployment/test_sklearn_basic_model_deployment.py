import pytest
from simple_logger.logger import get_logger
from typing import Any
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from utilities.constants import KServeDeploymentType, Protocols
from tests.model_serving.model_runtime.mlserver.constant import SKLEARN_REST_INPUT_QUERY, SKLEARN_GRPC_INPUT_QUERY, MODEL_PATH_PREFIX
from tests.model_serving.model_runtime.mlserver.constant import BASE_RAW_DEPLOYMENT_CONFIG, BASE_SEVERRLESS_DEPLOYMENT_CONFIG
from tests.model_serving.model_runtime.mlserver.utils import validate_inference_request


LOGGER = get_logger(name=__name__)

MODEL_NAME: str = "sklearn-iris"

MODEL_VERSION: str = "v1.0.0"

MODEL_NAME_DICT: dict[str, str] = {  "name": MODEL_NAME  }

MODEL_STORAGE_URI_DICT: dict[str, str] = {  "model-dir": f"{MODEL_PATH_PREFIX}/sklearn"  }


pytestmark = pytest.mark.usefixtures("root_dir", "valid_aws_config", "mlserver_rest_serving_runtime_template", "mlserver_grpc_serving_runtime_template")


@pytest.mark.parametrize(
    "protocol, model_namespace, mlserver_inference_service, s3_models_storage_uri, serving_runtime",
    [
        pytest.param(
            {   "protocol_type": Protocols.REST },
            {   "name": "sklearn-iris-raw-rest" },
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                **MODEL_NAME_DICT,
            },
                MODEL_STORAGE_URI_DICT,
                BASE_RAW_DEPLOYMENT_CONFIG,
            id="sklearn-iris-raw-rest-deployment",
        ),
        pytest.param(
            {   "protocol_type": Protocols.GRPC },
            {   "name": "sklearn-iris-raw-grpc" },
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                **MODEL_NAME_DICT,
            },
                MODEL_STORAGE_URI_DICT,
                BASE_RAW_DEPLOYMENT_CONFIG,
            id="sklearn-iris-raw-grpc-deployment",
        ),
        pytest.param(
            {   "protocol_type": Protocols.REST },
            {   "name": "sklearn-iris-serverless-rest"  },
            {
                **BASE_SEVERRLESS_DEPLOYMENT_CONFIG,
                **MODEL_NAME_DICT
            },
                MODEL_STORAGE_URI_DICT,
                BASE_SEVERRLESS_DEPLOYMENT_CONFIG,
            id="sklearn-iris-serverless-rest-deployment",
        ),
        pytest.param(
            {   "protocol_type": Protocols.GRPC },
            {   "name": "sklearn-iris-serverless-grpc"  },
            {
                **BASE_SEVERRLESS_DEPLOYMENT_CONFIG,
                **MODEL_NAME_DICT
            },
                MODEL_STORAGE_URI_DICT,
                BASE_SEVERRLESS_DEPLOYMENT_CONFIG,
            id="sklearn-iris-serverless-grpc-deployment",
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
        protocol: str,
        root_dir: str
    ):
        input_query = SKLEARN_REST_INPUT_QUERY if protocol == Protocols.REST else SKLEARN_GRPC_INPUT_QUERY

        validate_inference_request(
            pod_name=mlserver_pod_resource.name,
            isvc=mlserver_inference_service,
            response_snapshot=response_snapshot,
            input_query=input_query,
            model_version=MODEL_VERSION,
            protocol=protocol,
            root_dir=root_dir
        )