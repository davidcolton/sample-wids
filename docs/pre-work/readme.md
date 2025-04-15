# Pre-work

The labs in the workshop are [Jupyter notebooks](https://jupyter.org/). These notebooks are intended to be run locally on your computer. Follow these instructions to ensure that your laptop is ready to run the labs.

## Running the Granite Notebooks Locally

It is recommended if you want to run the lab notebooks locally on your computer that you have:

- A computer or laptop
- Knowledge of [Git](https://git-scm.com/) and [Python](https://www.python.org/)

Running the lab notebooks locally on your computer requires the following steps:

- [Local Prerequisites](#local-prerequisites)
- [Clone the Granite Workshop Repository](#clone-the-granite-workshop-repository)
- [Serving the Granite AI Models](#serving-the-granite-ai-models)

## Local Prerequisites

- Git
- Uv



### Git

Git can be installed on the most common operating systems like Windows,  Mac, and Linux. In fact, Git comes installed by default on most Mac and  Linux machines!

For comprehensive instructions on how to install `git` on your laptop please refer to the [Install Git](https://github.com/git-guides/install-git) page.

To confirm the you have `git` installed correctly you can open a terminal window and type `git version`. You should receive a response like the one shown below.

```shell
git version
git version 2.39.5 (Apple Git-154)
```



### Uv

`uv` is an extremely fast Python package and project manager, written in Rust.

For detailed instructions on how to install `uv` on your laptop please refer to the [Installing uv](https://docs.astral.sh/uv/getting-started/installation/#installing-uv) page.

To confirm the you have `uv` installed correctly you can open a terminal window and type `uv --version`. You should receive a response like the one shown below.

```shell
uv --version
uv 0.6.12 (e4e03833f 2025-04-02)
```



## Clone the Granite Workshop Repository

Clone the workshop repo and cd into the repo directory.

```shell
git clone https://github.com/davidcolton/sample-wids.git
cd sample-wids
```



### Sync the Python Virtual Environment

The Sample WiDS repository uses a `pyproject.toml` file to define the version of Python to use and the required libraries to load. To sync your repository and setup Python and download your library dependancies run `uv sync` in a terminal. After syncing you have to activate your virtual environment.

**Note:**

If running on Windows it is suggested that you use the Windows Powershell running as administrator or, if you have it installed, the Windows Subsystem for Linux.

```shell
uv sync

# Mac & Linux
source .venv/bin/activatez

# Windows Powershell
.venv\Scripts\activate
```

### Serving the Granite AI Models

[Lab 1: Document Summarization with Granite](../lab-1/readme.md), [Lab 2: Retrieval Augmented Generation (RAG) with Langchain](../lab-2/readme.md) and [Lab 4: Generating Bash Code with Granite Code](../lab-4/readme.md) require Granite models to be served by an AI model runtime so that the models can be invoked or called. There are 2 options to serve the models as follows:

- [Replicate AI Cloud Platform](#replicate-ai-cloud-platform)
- [Running Ollama Locally](#running-ollama-locally) OR

#### Replicate AI Cloud Platform

[Replicate](https://replicate.com/) is a cloud platform that will host and serve AI models for you.

1. Create a [Replicate](https://replicate.com/) account. You will need a [GitHub](https://github.com/) account to do this.

1. Add credit to your Replicate Account (optional). To remove a barrier to entry to try the Granite models on the Replicate platform, use [this link](https://replicate.com/invites/a8717bfe-2f3d-4a52-88ed-1356231cdf03) to add a small amount of credit to your Replicate account.

1. Create a Replicate [API Token](https://replicate.com/account/api-tokens).

1. When you run the sample Notebooks you will be prompted to enter this token.

1. Alternatively you can set your Replicate API Token as an environment variable in your terminal where you will run the notebook:

    ```shell
    export REPLICATE_API_TOKEN=<your_replicate_api_token>
    ```

#### Running Ollama Locally

If you want to run the AI models locally on your computer, you can use [Ollama](https://ollama.com/). You will need to have a computer with:

- GPU processor
- At least 32GB RAM

!!! note "Tested system"
    This was tested on a Macbook with an M1 processor and 32GB RAM. It maybe possible to serve models with a CPU and less memory.

If you computer is unable to serve the models, then it is recommended to go to the [Replicate AI Cloud Platform](#replicate-ai-cloud-platform) section instead.

Running Ollama locally on your computer requires the following steps:

1. [Download and install Ollama](https://github.com/ollama/ollama?tab=readme-ov-file#ollama), if you haven't already. **Ollama v0.3.14+ is required, so please upgrade if on an earlier version.**

    On macOS, you can use Homebrew to install with

    ```shell
    brew install ollama
    ```

1. Start the Ollama server. You will leave this running during the workshop.

    ```shell
    ollama serve
    ```

1. In another terminal window, pull down the Granite models you will want to use in the workshop. Larger models take more memory to run but can give better results.

    ```shell
    ollama pull granite3.2:2b
    ollama pull granite3.2:8b
    ```

    For [Lab 4: Generating Bash Code with Granite Code](../lab-4/readme.md), you will also need at least one of the following Granite Code models.

    ```shell
    ollama pull granite-code:3b
    ollama pull granite-code:8b
    ```
