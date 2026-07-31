import boto3
import logging

from shared.env import get_env

logger = logging.getLogger(__name__)

# Module-level cache to optimize execution speed during Lambda warm starts
_SECRETS_CACHE = {}

def get_secret(parameter_name: str) -> str:
    """
    Retrieves secure strings from AWS SSM Parameter Store.
    Provides a fallback to local environment variables for container testing.
    """
    if parameter_name in _SECRETS_CACHE:
        return _SECRETS_CACHE[parameter_name]

    # Local fallback priority
    local_value = get_env(parameter_name)
    if local_value:
        _SECRETS_CACHE[parameter_name] = local_value
        return local_value

    try:
        logger.info(f"Retrieving parameter token '{parameter_name}' from AWS SSM Store...")
        # Lambda injects AWS_REGION automatically; ap-south-1 is the deployment
        # default for anything running outside it.
        region = get_env("AWS_REGION", "ap-south-1")
        ssm_client = boto3.client("ssm", region_name=region)
        response = ssm_client.get_parameter(Name=parameter_name, WithDecryption=True)
        
        secret_value = response["Parameter"]["Value"]
        _SECRETS_CACHE[parameter_name] = secret_value
        return secret_value
        
    except Exception as e:
        logger.error(f"Critical failure retrieving parameter '{parameter_name}': {str(e)}")
        raise RuntimeError(f"System failed to initialize secure credential: {parameter_name}")