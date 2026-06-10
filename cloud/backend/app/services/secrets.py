import os
import boto3
import logging

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
    local_value = os.getenv(parameter_name)
    if local_value:
        _SECRETS_CACHE[parameter_name] = local_value
        return local_value

    try:
        logger.info(f"Retrieving parameter token '{parameter_name}' from AWS SSM Store...")
        ssm_client = boto3.client("ssm", region_name="ap-south-1")
        response = ssm_client.get_parameter(Name=parameter_name, WithDecryption=True)
        
        secret_value = response["Parameter"]["Value"]
        _SECRETS_CACHE[parameter_name] = secret_value
        return secret_value
        
    except Exception as e:
        logger.error(f"Critical failure retrieving parameter '{parameter_name}': {str(e)}")
        raise RuntimeError(f"System failed to initialize secure credential: {parameter_name}")