from dagster import Definitions, load_assets_from_modules, ScheduleDefinition, define_asset_job

from orchestration.orchestration import assets  # noqa: TID252

all_assets = load_assets_from_modules([assets])

vectorstore_job = define_asset_job(
    name="vectorstore_job",
    selection=["faiss_output_path"]
)

weekly_schedule = ScheduleDefinition(
    job=vectorstore_job,
    cron_schedule="0 0 * * 0"
)

defs = Definitions(
    assets=all_assets,
    jobs=[vectorstore_job],
    schedules=[weekly_schedule],
)