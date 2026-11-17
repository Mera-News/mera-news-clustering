# mera-news-clustering

The clustering job behind Mera News, by Mera Labs B.V. It groups news articles that report the same story. It reads the embedded articles that mera-news-data-server stores in MongoDB, clusters them with HDBSCAN, and writes the clusters back. The same code runs on a CPU, or on an NVIDIA GPU. It is licensed under the AGPL-3.0.

## How it fits

The data server reads feeds, creates embeddings and stores articles. This job reads those articles from the database and writes clusters next to them. It runs once and exits, so it is meant to be started on a schedule, for example every hour. The job does not call the data server and the data server does not call the job: they share the database. mera-news-data-server then uses the clusters to link topics to stories, build headline editions, and return each article's cluster membership from its news API.

| Collection | What the job does |
|---|---|
| `news-article` | Reads `_id`, `article_url`, `createdAt` and `embedding_article_data_text_matching`. Sets `processingStatus` to `aggregation_successful`, and sets `aggregationCompletedAt` and `updatedAt`, on every article in the window. |
| `news-cluster` | Writes one document per cluster: `_id`, `clusterSize`, `clusteringRunId`, `createdAt`, `updatedAt`, and `stableClusterId` for clusters of two or more articles. |
| `cluster-article-link` | Writes one document per article in a cluster: `_id`, `clusterId`, `articleId`, `confidence`, `createdAt`, `clusteringRunId`. |

The schemas live in mera-news-data-server (`libs/mera-models/src/models/`). The file `tests/contract/collections.json` in this repository states the shapes the job shares with it, and the two must change together.

## What you need

- The MongoDB database that your mera-news-data-server uses. See the data server's README for how to set it up.
- Docker, or Python 3.14.
- No GPU.

## Run it with Docker

```
docker build -f Dockerfile.clustering -t mera-news-clustering .
docker run --rm -e MONGODB_URI="mongodb+srv://user:password@host/database" -e CLUSTERING_GPU=false mera-news-clustering
```

`MONGODB_URI` must name the database (the part after the host, here `database`). The job uses the database named in the URI. Without one it cannot open the database, retries for about two minutes, and exits with an error.

The job runs once and exits. Start it from your own scheduler, such as cron or a systemd timer, and let one run finish before the next starts. The job has no lock of its own outside Google Cloud Run Jobs (see `CHAIN_GCP_PROJECT` below).

The image is linux/amd64 and large, about 6 to 8 GB, because it carries the GPU libraries even when it runs on a CPU. On another architecture, build it with `docker buildx build --platform=linux/amd64`.

## Run it from source (CPU only)

```
make setup
MONGODB_URI="mongodb+srv://user:password@host/database" make run-cpu
```

`make setup` needs `python3.14` on your path, creates `.venv` and installs `requirements-dev.txt`. `requirements.txt` alone is enough to run the job. The job needs Python 3.14 because the default `MONGODB_COMPRESSORS=zstd` uses the zstd module of the 3.14 standard library.

## GPU (optional)

Running on a GPU needs a linux/amd64 host with an NVIDIA driver for CUDA 12, and `CLUSTERING_GPU=true`, which is the default in the Docker image. The image then starts the job through `python -m cuml.accel`, which runs the same HDBSCAN code on the GPU. With `CLUSTERING_GPU=true` and no GPU the job fails at boot.

The GPU image installs NVIDIA's CUDA libraries from PyPI. Those libraries are proprietary and come under NVIDIA's own licence terms, not the AGPL. This repository's code is AGPL-3.0-only, and the CPU install (`make setup`, or `requirements.txt`) uses none of NVIDIA's libraries. The Docker image as built here installs them even when run with `CLUSTERING_GPU=false`.

## Configuration

Settings are environment variables, read in `src/config.py`. Names are case sensitive.

| Variable | Default | What it does |
|---|---|---|
| `MONGODB_URI` | empty | Connection string. Must include the database name. |
| `MONGODB_MAX_POOL_SIZE` | `1` | Size of the driver's connection pool. Raise it for `DB_WRITE_CONCURRENCY` to take effect. |
| `MONGODB_READ_PREFERENCE` | `primary` | Read preference for the window read. |
| `MONGODB_COMPRESSORS` | `zstd` | Wire compression. Empty disables it. The job refuses to start if the driver cannot use the compressor. |
| `CLUSTER_WINDOW_HOURS` | `48` | How far back the job reads articles. |
| `MIN_CLUSTER_SIZE` | `2` | HDBSCAN `min_cluster_size`. |
| `MIN_SAMPLES` | `1` | HDBSCAN `min_samples`. |
| `CLUSTER_SELECTION_METHOD` | `eom` | HDBSCAN `cluster_selection_method`. |
| `CLUSTER_SELECTION_EPSILON` | `0.0` | HDBSCAN `cluster_selection_epsilon`. A value of 0 or less leaves it at the HDBSCAN default. |
| `CLUSTERING_GPU` | `false` in code, `true` in the Docker image | In the Docker image, `true` starts the job under `cuml.accel` and needs a GPU. Outside the image it is only reported in the logs. |
| `PERSIST_SINGLETON_CLUSTERS` | `true` | When `false`, articles that match nothing get no cluster and no link. They are still stamped. |
| `PERSIST_LEGACY_WIPE_FIRST` | `false` | When `true`, deletes both cluster collections before inserting, instead of the generation swap described below. |
| `STABLE_MIN_SHARED` | `2` | Articles a new cluster must share with a previous one to inherit its `stableClusterId`. |
| `STABLE_MIN_OVERLAP_RATIO` | `0.5` | The overlap must also be at least this share of the smaller of the two clusters. |
| `FETCH_BATCH_SIZE` | `2000` | Cursor batch size for the window read. |
| `FETCH_PARALLELISM` | `4` | Number of `createdAt` ranges the window read fetches in parallel. |
| `DB_WRITE_BATCH_SIZE` | `2000` | Documents per insert. |
| `DB_WRITE_CONCURRENCY` | `4` | Parallel insert workers, capped by `MONGODB_MAX_POOL_SIZE`. |
| `DB_WRITE_PAUSE_S` | `0.0` | A value above 0 switches to sequential inserts with this pause between batches. |
| `STAMP_BATCH_SIZE` | `10000` | Articles per update when stamping the window. |
| `LOG_LEVEL` | `info` | `info`, `warn` or `error`. |
| `CHAIN_GCP_PROJECT` | empty | Google Cloud project for the Cloud Run Jobs singleton guard. |
| `CHAIN_GCP_REGION` | `europe-west1` | Region for the same guard. |
| `CLOUD_RUN_JOB`, `CLOUD_RUN_EXECUTION` | unset | Set by Cloud Run Jobs. |
| `JOB_ENTRYPOINT` | `src.job_main_mongo` | Docker image only: the module the image runs. |

The singleton guard lists the job's executions through the Cloud Run Jobs API and exits quietly if an older execution is still running. It does nothing unless `CLOUD_RUN_JOB`, `CLOUD_RUN_EXECUTION`, `CHAIN_GCP_PROJECT` and `CHAIN_GCP_REGION` are all set.

Mera runs `CLUSTER_SELECTION_EPSILON=0.30` and `PERSIST_SINGLETON_CLUSTERS=false`. The defaults are `0.0` and `true`.

## How it works

1. The job reads the articles created in the last 48 hours that have a non-empty `embedding_article_data_text_matching` vector.
2. Articles with the same `article_url` are collapsed to one, keeping the newest.
3. Each vector is normalised to unit length (L2). HDBSCAN then runs with `min_cluster_size=2`, `min_samples=1`, the `eom` selection method and euclidean distance. On unit vectors, euclidean distance orders pairs the same way as cosine distance.
4. Each article HDBSCAN leaves out as noise becomes a cluster of one.
5. Clusters of two or more articles keep a `stableClusterId` across runs. A new cluster inherits the id of a previous cluster when they share enough members (`STABLE_MIN_SHARED`, `STABLE_MIN_OVERLAP_RATIO`). Otherwise it gets a new id.
6. The new clusters and links are inserted first, then the documents of earlier runs are deleted by `clusteringRunId`. Readers never see an empty collection during a run.
7. Every article in the window is stamped.

Known limit: one story often ends up split across several clusters.

## Logs

The job writes one JSON object per line to stdout. The levels are `info` for the run narrative, `warn` for recoverable problems such as a database retry, and `error` for failures that stop the run. There is no `debug` level.

The line with the message `run_summary` closes a run. Its main fields are `windowArticles`, `clusters`, `multiArticleClusters`, `singletons`, `stableCarried`, `stableMinted`, `maxClusterSize`, `clusteringRunId` and the stage timings (`fetchMs`, `fitMs`, `persistMs`, `durationMs`).

The job exits non-zero only when the infrastructure fails: the database is unreachable or a read or write fails, the configuration cannot be used (for example a compressor the driver cannot load), or the clustering library does not start. An empty window is a normal exit.

## Checking a dependency upgrade

`src.equivalence_main` tells you whether a dependency upgrade changed the clustering. It fits the same window with fixed settings and stores the labels in the `ml-equivalence-run` collection. Run it against a copy of the data, not the live database.

1. With the old image or environment, set `EQUIV_TAG=a` and `EQUIV_WINDOW_END` to a fixed timestamp such as `2026-10-31T12:00:00+00:00`, and run `python -m src.equivalence_main`.
2. With the new one, run the same with `EQUIV_TAG=b` and the same `EQUIV_WINDOW_END`.
3. Run it with `EQUIV_COMPARE=a,b`. The `equivalence_verdict` log line says `IDENTICAL` or `DIFFERENT`.
4. Run it with `EQUIV_DROP=true` to empty the collection.

The other settings are `EQUIV_EPSILONS` (default `0,0.30`), `EQUIV_MAX_ROWS` (default `0`, the whole window) and `EQUIV_SEED` (default `17`). It never writes clusters and never stamps articles.

## Development

```
make test
.venv/bin/ruff check .
```

The pure tests need nothing. The integration tests start a real `mongod`: set `MONGOD_BIN` to the path of a MongoDB 8.x server binary, or put `mongod` on your path. Tests run with coverage, and the gate is in `pyproject.toml`. The image end-to-end test and the licence check run in CI.

## License

AGPL-3.0-only. See [LICENSE](LICENSE). If the AGPL does not fit your use, see [COMMERCIAL.md](COMMERCIAL.md). The names and logos are covered by [TRADEMARK.md](TRADEMARK.md).

"Mera", "Mera News" and the Mera logos are trademarks of Mera Labs B.V., a company based in Amsterdam, the Netherlands. Nobody may ship a fork, a modified build or a hosted service based on this code under the name Mera or Mera News.

Mera Labs B.V. can offer a commercial license to organisations that cannot meet the AGPL's terms. To ask about one, write to contact@mera.news.

## Contributing

Issues are welcome. Code contributions are not accepted yet.
