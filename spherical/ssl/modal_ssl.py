"""Run an FR-ROT or RFF-FR-ROT SSL ``unif_w`` sweep on one Modal A100-40GB.

Usage:
    modal run --detach modal_ssl.py
    modal run --detach modal_ssl.py --unif-ws "0.25,0.5,1,2,5,10"
    modal run --detach modal_ssl.py --unif-ws "1,2" --seed 1 --rho 0.005

Persistent artifacts are stored in the Modal Volumes
``fr-rot-cifar10`` and ``fr-rot-ssl-results``.
"""

from __future__ import annotations

import modal


APP_NAME = "fr-rot-ssl-sweep"
REPO_URL = "https://github.com/tuanhacs/fr_rot.git"
REPO_DIR = "/root/fr_rot"
SSL_DIR = f"{REPO_DIR}/spherical/ssl"
DATA_DIR = "/data"
RESULTS_DIR = "/results"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "libglib2.0-0", "libgl1")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "numpy==1.26.4",
        "scipy==1.15.3",
        "scikit-learn==1.7.2",
        "POT==0.9.6.post1",
        "geotorch==0.3.0",
        "fast-pytorch-kmeans==0.2.2",
        "pandas==2.3.3",
        "matplotlib==3.10.7",
        "seaborn==0.13.2",
        "tqdm==4.67.1",
        "Pillow==12.0.0",
    )
    # Re-run the clone layer so a new invocation picks up the latest commit.
    .run_commands(
        f"git clone --depth 1 {REPO_URL} {REPO_DIR}",
        force_build=True,
    )
    .run_commands(
        f"python -m pip install --no-cache-dir -e {REPO_DIR}/power_spherical"
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
        }
    )
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name("fr-rot-cifar10", create_if_missing=True)
results_volume = modal.Volume.from_name(
    "fr-rot-ssl-results", create_if_missing=True
)


@app.function(
    gpu="A100-40GB",
    cpu=4,
    memory=4096,
    timeout=24 * 60 * 60,
    volumes={
        DATA_DIR: data_volume,
        RESULTS_DIR: results_volume,
    },
)
def train_sweep(
    unif_ws: list[float],
    seed: int = 0,
    rho: float = 0.005,
    fiber_tau: float = 1.0,
    method: str = "fr_rot",
    num_frequencies: int = 1,
    rff_sigma: float = 1.0,
) -> None:
    import subprocess

    for unif_w in unif_ws:
        run_suffix = f"_modal_{method}_unif_{unif_w:g}_rho_{rho:g}_seed_{seed}"
        command = [
            "python",
            "train_eval.py",
            "--method",
            method,
            "--data_folder",
            DATA_DIR,
            "--result_folder",
            RESULTS_DIR,
            "--identifier",
            run_suffix,
            "--ntrees",
            "200",
            "--nlines",
            "20",
            "--delta",
            "2",
            "--unif_w",
            str(unif_w),
            "--feat_dim",
            "10",
            "--epochs",
            "200",
            "--batch_size",
            "512",
            "--momentum",
            "0.9",
            "--weight_decay",
            "1e-3",
            "--lr",
            "0.05",
            "--seed",
            str(seed),
            "--p",
            "1",
            "--p_agg",
            "1",
            "--rho",
            str(rho),
            "--fiber_tau",
            str(fiber_tau),
            "--num_frequencies",
            str(num_frequencies),
            "--rff_sigma",
            str(rff_sigma),
        ]

        print(f"\n{'=' * 80}")
        print(
            f"Starting {method} SSL: unif_w={unif_w:g}, rho={rho:g}, "
            f"fiber_tau={fiber_tau:g}, K={num_frequencies}, "
            f"rff_sigma={rff_sigma:g}, seed={seed}"
        )
        print(" ".join(command))
        print(f"{'=' * 80}\n")

        subprocess.run(command, cwd=SSL_DIR, check=True)
        data_volume.commit()
        results_volume.commit()


@app.local_entrypoint()
def main(
    unif_ws: str = "0.5,1,2,3,4,5,6,7,8,9,10",
    seed: int = 0,
    rho: float = 0.005,
    fiber_tau: float = 1.0,
    method: str = "fr_rot",
    num_frequencies: int = 1,
    rff_sigma: float = 1.0,
) -> None:
    weights = [float(value.strip()) for value in unif_ws.split(",") if value.strip()]
    if not weights:
        raise ValueError("unif_ws must contain at least one numeric value")
    if any(weight < 0 for weight in weights):
        raise ValueError("unif_w values must be non-negative")
    if method not in ("fr_rot", "rff_fr_rot"):
        raise ValueError("method must be 'fr_rot' or 'rff_fr_rot'")

    call = train_sweep.spawn(
        weights, seed, rho, fiber_tau, method, num_frequencies, rff_sigma
    )
    print(f"Submitted one sequential {method} SSL sweep: {call.object_id}")
    print("The same A100-40GB container will run every unif_w in order.")
