from invoke import task
import os
import sys

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()


def check_venv():
    """Check if a virtual environment, Conda environment, or uv environment is active."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    venv_env = os.environ.get("VIRTUAL_ENV")

    if conda_env:
        print(f"You are in a Conda environment: {conda_env}")
        return

    # Classic venv detection
    if venv_env or (
        hasattr(sys, "base_prefix")
        and sys.base_prefix != getattr(sys, "prefix", sys.base_prefix)
    ):
        print(f"You are running in a Python virtual environment: {venv_env or sys.prefix}")
        return

    # uv detection: look for a .venv folder (created by `uv venv`)
    if os.path.exists(".venv"):
        print("Detected uv environment (.venv). Using uv to run commands.")
        return

    print("No virtual environment or Conda environment detected. Please create and activate one first.")
    sys.exit(1)


@task
def install(c):
    """Install dependencies using uv and pip."""
    check_venv()
    c.run("pip install uv")
    c.run("uv pip install -r requirements.txt")


##########################################
# Air Quality Batch ML System
##########################################

@task
def aq_clean(c):
    check_venv()
    c.run("uv run --from .venv python mlfs/clean_hopsworks_resources.py aq")

@task
def aq_backfill(c):
    check_venv()
    c.run("ipython notebooks/airquality/1_air_quality_feature_backfill.ipynb")

@task
def aq_features(c):
    check_venv()
    c.run("ipython notebooks/airquality/2_air_quality_feature_pipeline.ipynb")

@task
def aq_train(c):
    check_venv()
    c.run("ipython notebooks/airquality/3_air_quality_training_pipeline.ipynb")

@task
def aq_inference(c):
    check_venv()
    c.run("ipython notebooks/airquality/2_air_quality_feature_pipeline.ipynb")
    c.run("ipython notebooks/airquality/4_air_quality_batch_inference.ipynb")


@task
def aq_llm(c):
    check_venv()
    c.run("ipython notebooks/airquality/5_function_calling.ipynb")

@task(pre=[aq_backfill, aq_features, aq_train, aq_inference])
def aq_all(c):
    pass



@task
def install_recommender(c):
    """Install LLM-related dependencies."""
    check_venv()
    c.run("uv pip install -r requirements-llm.txt")


##########################################
# Credit Card Fraud Real-Time ML System
##########################################

@task
def cc_start_ui(c):
    """Start Streamlit app."""
    check_venv()
    c.run("uv run python -m streamlit run streamlit_app.py", pty=True)


@task
def cc_clean(c):
    """Clean Hopsworks resources."""
    check_venv()
    c.run("uv run python mlfs/clean_hopsworks_resources.py cc", pty=True)


@task
def cc_datagen(c):
    """Run data generation notebook."""
    check_venv()
    c.run("ipython notebooks/ccfraud/0-data-generation-with-polars.ipynb", pty=True)


@task
def cc_gen_kafka(c):
    """Run Kafka transactions generator notebook."""
    check_venv()
    c.run("ipython notebooks/ccfraud/transactions_synthetic_kafka_generator.ipynb", pty=True)


@task
def cc_features(c):
    """Run batch Polars feature pipeline."""
    check_venv()
    c.run("ipython notebooks/ccfraud/1-batch-polars-feature-pipeline.ipynb", pty=True)


@task
def cc_streaming_features(c):
    """Run streaming feature pipeline with Feldera."""
    check_venv()
    c.run("ipython notebooks/ccfraud/1-streaming-feature-pipeline-feldera.ipynb", pty=True)


@task
def cc_train(c):
    """Run training notebook."""
    check_venv()
    c.run("ipython notebooks/ccfraud/", pty=True)


@task
def cc_deploy(c):
    """Run deployment notebook."""
    check_venv()
    c.run("ipython notebooks/ccfraud/", pty=True)


##########################################
# Titanic ML System
##########################################

@task
def titanic_clean(c):
    check_venv()
    c.run("uv run python mlfs/clean_hopsworks_resources.py ")

@task
def titanic_features(c):
    check_venv()
    c.run("ipython  notebooks/titanic/1-titanic-feature-group-backfill.ipynb")

@task
def titanic_train(c):
    check_venv()
    c.run("ipython notebooks/titanic/2-titanic-training-pipeline.ipynb")

@task
def titanic_inference(c):
    check_venv()
    c.run("ipython notebooks/titanic/scheduled-titanic-feature-pipeline-daily.ipynb")
    c.run("ipython notebooks/titanic/scheduled-titanic-batch-inference-daily.ipynb")


@task(pre=[titanic_features, titanic_train, titanic_inference])
def titanic_all(c):
    pass



