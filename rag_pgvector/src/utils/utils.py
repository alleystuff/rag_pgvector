from azure.identity import DefaultAzureCredential, get_bearer_token_provider


def get_token_provider():
    return get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )


def notebook_line_magic():
    from IPython import get_ipython
    ip = get_ipython()
    ip.run_line_magic("reload_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")