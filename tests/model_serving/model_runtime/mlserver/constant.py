import os
from typing import Any, Union
from utilities.constants import Protocols, KServeDeploymentType, RuntimeTemplates

MODEL_PATH_PREFIX: str = "mlserver/model_repository"

INPUT_QUERY: dict[str, Any] = {"inputs":[{"name":"input-0","shape":[2,4],"datatype":"FP32","data":[[6.8,2.8,4.8,1.4],[6,3.4,4.5,1.6]]}]}

TEMPLATE_FILE_PATH: dict[str, str] = {
    Protocols.REST: os.path.join(os.path.dirname(__file__), "mlserver_rest_serving_template.yaml"),
    Protocols.GRPC: os.path.join(os.path.dirname(__file__), "mlserver_grpc_serving_template.yaml"),
}

TEMPLATE_MAP: dict[str, str] = {
    Protocols.REST: RuntimeTemplates.MLSERVER_REST,
    Protocols.GRPC: RuntimeTemplates.MLSERVER_GRPC,
}

PREDICT_RESOURCES: dict[str, Union[list[dict[str, Union[str, dict[str, str]]]], dict[str, dict[str, str]]]] = {
    "volumes": [
        {"name": "shared-memory", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
        {"name": "tmp", "emptyDir": {}},
        {"name": "home", "emptyDir": {}},
    ],
    "volume_mounts": [
        {"name": "shared-memory", "mountPath": "/dev/shm"},
        {"name": "tmp", "mountPath": "/tmp"},
        {"name": "home", "mountPath": "/home/mlserver"},
    ],
    "resources": {"requests": {"cpu": "2", "memory": "15Gi"}, "limits": {"cpu": "3", "memory": "16Gi"}},
}

BASE_RAW_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
    "min-replicas": 1,
}

BASE_SEVERRLESS_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_mode": KServeDeploymentType.SERVERLESS,
    "min-replicas": 1,
}
